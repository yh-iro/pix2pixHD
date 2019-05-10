### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,d_real,d_fake),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        self.noise_for_fake_img = Variable(torch.zeros(opt.dic['batchSize'], opt.dic['input_nc'], opt.dic['loadSize'], opt.dic['loadSize']).cuda())
        self.noise_for_real_img = Variable(torch.zeros(opt.dic['batchSize'], opt.dic['input_nc'], opt.dic['loadSize'], opt.dic['loadSize']).cuda())
        self.std_max = opt.dic['std_max']

        BaseModel.initialize(self, opt)
        if opt.dic['resize_or_crop'] != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.dic['instance_feat'] or opt.dic['label_feat']
        self.gen_features = self.use_features and not self.opt.dic['load_features']
        input_nc = opt.dic['label_nc'] if opt.dic['label_nc'] != 0 else opt.dic['input_nc']

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        if not opt.dic['no_instance']:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.dic['feat_num']
        self.netG = networks.define_G(netG_input_nc, opt.dic['output_nc'], opt.dic['ngf'], opt.dic['netG'],
                                      opt.dic['n_downsample_global'], opt.dic['n_blocks_global'], opt.dic['n_local_enhancers'],
                                      opt.dic['n_blocks_local'], opt.dic['norm'], gpu_ids=self.gpu_ids)

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.dic['no_lsgan']
            netD_input_nc = input_nc + opt.dic['output_nc']
            if not opt.dic['no_instance']:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.dic['ndf'], opt.dic['n_layers_D'], opt.dic['norm'], use_sigmoid,
                                          opt.dic['num_D'], not opt.dic['no_ganFeat_loss'], gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.dic['output_nc'], opt.dic['feat_num'], opt.dic['nef'], 'encoder',
                                          opt.dic['n_downsample_E'], norm=opt.dic['norm'], gpu_ids=self.gpu_ids)
        if self.opt.dic['verbose']:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.dic['continue_train'] or opt.dic['load_pretrain']:
            pretrained_path = '' if not self.isTrain else opt.dic['load_pretrain']
            self.load_network(self.netG, 'G', opt.dic['which_epoch'], pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.dic['which_epoch'], pretrained_path)
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.dic['which_epoch'], pretrained_path)

        # set loss functions and optimizers
        if self.isTrain:
            if opt.dic['pool_size'] > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.dic['pool_size'])
            self.old_lr = opt.dic['lr']

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.dic['no_ganFeat_loss'], not opt.dic['no_vgg_loss'])
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.dic['no_lsgan'], tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.dic['no_vgg_loss']:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
        
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.dic['niter_fix_global'] > 0:
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.dic['n_local_enhancers'])):
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.dic['niter_fix_global'])
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.dic['lr'], betas=(opt.dic['beta1'], 0.999))

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.dic['lr'], betas=(opt.dic['beta1'], 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):             
        if self.opt.dic['label_nc'] == 0:
            if torch.cuda.is_available():
                input_label = label_map.data.cuda()
            else:
                input_label = label_map.data
        else:
            # create one-hot vector for label map 
            size = label_map.size()
            oneHot_size = (size[0], self.opt.dic['label_nc'], size[2], size[3])
            input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
            input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
            if self.opt.dic['data_type'] == 16:
                input_label = input_label.half()

        # get edges from instance map
        if not self.opt.dic['no_instance']:
            if torch.cuda.is_available():
                inst_map = inst_map.data.cuda()
            else:
                inst_map = inst_map.data
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)
        with torch.no_grad():
            input_label = Variable(input_label)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())

        # instance map for feature encoding
        if self.use_features:
            # get precomputed feature maps
            if self.opt.dic['load_features']:
                feat_map = Variable(feat_map.data.cuda())
            if self.opt.dic['label_feat']:
                inst_map = label_map.cuda()

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, infer=False):
        # Encode Inputs
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)
        self.noise_for_real_img.data.normal_(0, std=np.random.uniform(high=self.std_max))
        real_image += self.noise_for_real_img

        # Fake Generation
        if self.use_features:
            if not self.opt.dic['load_features']:
                feat_map = self.netE.forward(real_image, inst_map)                     
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label
        self.noise_for_fake_img.data.normal_(0, std=np.random.uniform(high=self.std_max))
        fake_image = self.netG.forward(input_concat, noise=self.noise_for_fake_img)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.dic['no_ganFeat_loss']:
            feat_weights = 4.0 / (self.opt.dic['n_layers_D'] + 1)
            D_weights = 1.0 / self.opt.dic['num_D']
            for i in range(self.opt.dic['num_D']):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.dic['lambda_feat']
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.dic['no_vgg_loss']:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.dic['lambda_feat']
        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake ), None if not infer else fake_image ]

    def inference(self, label, inst, image=None):
        # Encode Inputs        
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

        # Fake Generation
        if self.use_features:
            if self.opt.dic['use_encoded_image']:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features             
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)                        
        else:
            input_concat = input_label        
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                fake_image = self.netG.forward(input_concat)
        else:
            fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.dic['checkpoints_dir'], self.opt.dic['name'], self.opt.dic['cluster_path'])
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.dic['feat_num'], inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.dic['feat_num']):
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.dic['data_type']==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.dic['feat_num']
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.dic['label_nc']):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.dic['data_type']==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.dic['lr'], betas=(self.opt.dic['beta1'], 0.999))
        if self.opt.dic['verbose']:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.dic['lr'] / self.opt.dic['niter_decay']
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.dic['verbose']:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
