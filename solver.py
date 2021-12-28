import cv2
import gc 
import json
import matplotlib.pyplot as plt
import os
import time
import torch
import torch.nn.functional as F
from evaluation import *
from metrics import *
from network import U_Net, AttU_Net, DGAttU_Net, IWAttU_Net, IAWAttU_Net, SAU_Net, DamageNet, init_weights ##, STAU_Net
from torch import optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.utils import save_image

torch.multiprocessing.set_sharing_strategy('file_system')

class Solver(object):
  def __init__(self, config, ls_loader=None, ls_val_loader=None, test_loader=None):
    # Data loader
    self.ls_train = ls_loader
    self.ls_valid = ls_val_loader
    self.test_loader = test_loader

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Models
    ## self.adain = config.adain
    ## self.gram_similarity = None
    ## self.mean = None
    ## self.std = None
    self.model_type = config.model_type
    self.unet = None
    ## self.grad_init = config.grad_init
    self.init_type = config.init_type
    self.optim = None
    self.lr_scheduler = None
    self.lr_scheduler2 = None
    self.image_size = config.image_size
    self.image_ch = config.image_ch
    self.output_ch = config.output_ch
    self.result = None
    self.iters_to_accumulate = config.iters_to_accumulate
    self.meada = config.meada
    self.flags = config
    self.task_type = 'CF' ## classification

    # Hyper-parameters
    self.beta1 = config.beta1
    self.beta2 = config.beta2
    self.lr = config.lr
    self.threshold = config.threshold

    # Training settings
    self.epoch = 0
    self.warmup = config.warmup
    self.batch_size = config.batch_size
    self.multi_style = config.multi_style
    self.num_epochs = config.num_epochs
    
    # Criterion
    self.loss_type = config.loss_type
    if self.loss_type == 'Dice':
        self.criterion = DiceLoss(penalty_weight=config.loss_penalty_weight)
    if self.loss_type == 'SoftDice':
        self.criterion = SoftDiceLoss()
    elif self.loss_type == 'BCEDice':
        self.criterion = BCEDiceLoss(penalty_weight=config.loss_penalty_weight)
    elif self.loss_type == 'BCEJaccard':
        self.criterion = LossBinaryJaccard(jaccard_weight=config.loss_penalty_weight)
    elif self.loss_type == 'BCELovasz':
        self.criterion = BCELovaszLoss()
    elif self.loss_type == 'Focal':
        self.criterion = FocalLoss()

    # Path
    self.mode = config.mode
    self.model_to_load = config.model_to_load
    self.model_path = config.model_path
    self.test_path = config.test_dir
    self.inf_path = config.inf_folder
    self.val_root = config.val_infix
    try:
        os.makedirs(self.model_path)
    except FileExistsError:
        pass
    try:
        os.makedirs(self.val_root)
    except FileExistsError:
        pass
    
    self.unet, self.optim, self.cont_optim = self.build_model()
    
    ## learning rate
    self.lr_type = config.lr_scheduler
    if self.lr_type == 'Multi_MultiStep_LR':
        ## 3: no data parallel and no gradient accumulation
        ## self.lr_scheduler = MultiStepLR(self.optim, milestones=[5, 10, 20, 40], gamma=0.95)
        ## self.lr_scheduler2 = MultiStepLR(self.optim, milestones=[50, 60, 70, 80, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140], gamma=0.75)
        
        ## 5: data parallel and gradient accumulation
        self.lr_scheduler = MultiStepLR(self.optim, milestones=[10, 20, 30, 40], gamma=0.95)
        ## 6: deepglobe
        ## self.lr_scheduler = MultiStepLR(self.optim, milestones=[3, 5, 7, 10, 15, 20, 25, 30, 35, 40], gamma=0.95)

        self.lr_scheduler2 = MultiStepLR(self.optim, milestones=[50, 60, 70, 80, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140], gamma=0.75)
    ## lr warmup
    ## self.scheduler_warmup = GradualWarmupScheduler(self.optim, multiplier=1, total_epoch=10, after_scheduler=self.scheduler_steplr)

    # self.print_network(self.unet, self.model_type)

  def build_model(self):
    if self.model_type =='U_Net':
        unet = U_Net(img_ch=self.image_ch, output_ch=self.output_ch)
        ## unet = U_Net.Resnet34_upsample(num_classes=self.output_ch, num_channels=self.image_ch)
    ## elif self.model_type =='STAU_Net':
    ##     unet = STAU_Net(img_ch=self.image_ch, output_ch=self.output_ch, multi_style=self.multi_style, warmup=self.warmup)
    elif self.model_type == 'AttU_Net':
        unet = AttU_Net(img_ch=self.image_ch, output_ch=self.output_ch)
    elif self.model_type == 'DGAttU_Net':
        unet = DGAttU_Net(img_ch=self.image_ch, output_ch=self.output_ch)
    elif self.model_type == 'IWAttU_Net':
        unet = IWAttU_Net(img_ch=self.image_ch, output_ch=self.output_ch)
    elif self.model_type == 'IAWAttU_Net':
        unet = IAWAttU_Net(img_ch=self.image_ch, output_ch=self.output_ch)
    elif self.model_type == 'SAU_Net':
        unet = SAU_Net(img_ch=self.image_ch, output_ch=self.output_ch, cont_att=False)
    elif self.model_type == 'proposed':
        unet = IAWAttU_Net(img_ch=self.image_ch, output_ch=self.output_ch, cont_att=True)
    elif self.model_type == 'DamageNet':
        unet = DamageNet(img_ch=self.image_ch, output_ch=self.output_ch)
    
    ### weight initialization
    if 'training' in self.mode:
        init_weights(unet, init_type=self.init_type)
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            unet = nn.DataParallel(unet)
            print('data parallel on')
    unet.to(self.device)
    
    params_to_task = []
    ## params_to_norm = []
    params_to_cont = []
    for name, param in unet.named_parameters():
        ## if 'norm_conv' in name:
        ##     params_to_norm.append(param)
        if 'cont_att' in name:
            params_to_cont.append(param)
        else:
            params_to_task.append(param)
    
    optimizer = optim.Adam(params_to_task, self.lr, [self.beta1, self.beta2])
    ## optimizer = optim.AdamW(list(unet.parameters()), self.lr, [self.beta1, self.beta2])
    ## optimizer = optim.Adam(list(unet.parameters()), self.lr, [self.beta1, self.beta2])
    ## optimizer = optim.Adagrad(list(unet.parameters()), self.lr)
    
    if len(params_to_cont) > 0:
        cont_optimizer = optim.SGD(params_to_cont, lr=1e-4, weight_decay=0.9, momentum=0.9)
        print('cont_optimizer constructed')
    else:
        cont_optimizer = None
    return unet, optimizer, cont_optimizer

  def add_padding(self, image, image_size):
    ## img_list = []
    ## for img in image:
    ##     if img.shape[2] < image_size or img.shape[3] < image_size:
    ##       pad = True
    ##       pad_h = int((image_size-img.shape[2])/2)
    ##       pad_h2 = image_size - img.shape[2] - pad_h
    ##       pad_v = int((image_size - img.shape[3])/2)
    ##       pad_v2 = image_size - img.shape[3] - pad_v
    ##       img = F.pad(img, (pad_v, pad_v2, pad_h, pad_h2), value=0)
    ##       img_list.append((img, pad, pad_h, pad_h2, pad_v, pad_v2))
    ##     else:
    ##       img_list.append((img, False, 0, 0, 0, 0))
    ## return img_list
    if image.shape[-2] < image_size or image.shape[-1] < image_size:
      pad = True
      pad_h = int((image_size-image.shape[-2])/2)
      pad_h2 = image_size - image.shape[-2] - pad_h
      pad_v = int((image_size - image.shape[-1])/2)
      pad_v2 = image_size - image.shape[-1] - pad_v
      image = F.pad(image, (pad_v, pad_v2, pad_h, pad_h2), value=0)
      return image, pad, pad_h, pad_h2, pad_v, pad_v2
    else:
      return image, False, 0, 0, 0, 0
  
  def count_parameters(self, model):
      return sum(p.numel() for p in model.parameters() if p.requires_grad)

  def score_average(self, perf):
    perf['recall'] = perf['recall']/perf['length']
    perf['precision'] = perf['precision']/perf['length']
    perf['f1'] = perf['f1']/perf['length']
    perf['jaccard'] = perf['jaccard']/perf['length']
    perf['score'] = perf['jaccard'] * perf['f1']
    return perf

  def cal_perf(self, perf, sr, gt):
    ### SR is sigmoided
    SR = (sr > self.threshold).float()
    GT = gt
    ## SR_np = (sr > self.threshold).float().detach().view(-1).numpy()
    ## GT_np = gt.detach().view(-1).numpy()
    ## SR = SR_np.astype(int)
    ## GT = GT_np.astype(int)
    perf['recall'] += get_sensitivity(SR, GT)
    perf['precision'] += get_precision(SR, GT)
    perf['f1'] += get_F1(SR, GT)
    perf['jaccard'] += get_JS(SR, GT)
    ## perf['recall'] += recall_score(GT, SR, average='binary')
    ## perf['precision'] += precision_score(GT, SR, average='binary')
    ## perf['f1'] += f1_score(GT, SR, average='binary')
    ## perf['jaccard'] += jaccard_score(GT, SR, average='binary')
    perf['length'] += 1
    return perf

  def save_model(self, model, unet_path, perf, v_set, epoch, best_score):
    if perf['f1'] > best_score or best_score == 0.0:
        best_score = perf['f1']
        prefix = unet_path[:-3]+'_'+str(perf['f1'])

        ## save inferring images
        ## for n in range(len(v_set)):
        ##     res = torch.cat(v_set[n], dim=0)
        ##     save_image(res, f'{self.val_root}/{epoch}_valid_{n}.png', nrow=self.batch_size)
        if self.cont_optim is None:
            torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(), 'optim_state_dict': self.optim.state_dict(), 'scheduler': self.lr_scheduler.state_dict() }, prefix+'.pt')
        else:
            torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(), 'optim_state_dict': self.optim.state_dict(), 'cont_optim_state_dict': self.cont_optim.state_dict(), 'scheduler': self.lr_scheduler.state_dict() }, prefix+'.pt')

        print('Best %s model score(F1) : %.4f'%(self.model_type, best_score))
    return best_score

  def infer(self, imgs, gts, state='training'):
    img, post_img = imgs
    pad = False
    pad_h, pad_h2, pad_v, pad_v2 = 0, 0, 0, 0
    image, pad, pad_h, pad_h2, pad_v, pad_v2 = self.add_padding(img, self.image_size)
    if post_img is not None:
        post_image, pad, pad_h, pad_h2, pad_v, pad_v2 = self.add_padding(post_img, self.image_size)
    else:
        post_image = None
    
    ## image = img.to(self.device)
    ## if self.model_type == 'U_Net' or self.model_type == 'AttU_Net':
    sr, ground, output = self.unet((image, post_image), gts, state)

    ## saving feature maps
    ## if state == 'testing':
    ##     b, c, h, w = output['embedding'].size()
    ##     for i  in range(b):
    ##         fm = output['embedding'][i]
    ##         fm = fm.permute(1, 0, 2, 3)
    ##         print(f'fm size {fm.size()}')  ## 1024*1*32*32
    ##         save_image(output['embedding'], f'{self.val_root}/feature_map/featuremap_{i}.png', nrow=32)
    ## print(error)
     
    if pad:
        SR = F.pad(sr, (-pad_v, -pad_v2, -pad_h, -pad_h2))
        if 'd1_diff' in output.keys():
            output['d1_diff'] = F.pad(output['d1_diff'], (-pad_v, -pad_v2, -pad_h, -pad_h2))
            ### output['d1_cont'] = F.pad(output['d1_cont'], (-pad_v, -pad_v2, -pad_h, -pad_h2))
    else:
        SR = sr
    return SR, ground, output
    ## elif self.model_type == 'STAU_Net':
    ##     sr, ground, gram_loss = self.unet(image, gt, self.warmup)
    ##     return sr, ground, gram_loss

  def get_onehot(self, gt, N):
      gt[gt == -1] = 3    ## 0: not road, 3: recovered road, 2: damaged road, 1: unchanged road
      gt_onehot = F.one_hot(gt.long(), num_classes=4)
      ### indice = torch.tensor([1, 2, 3]).to(self.device)
      ## print(gt.size(), gt_onehot.size())  ## torch.Size([16, 250000]) torch.Size([16, 250000, 4])
      ## gt = gt.long().reshape(-1)
      ## ones = torch.sparse.torch.eye(N, device=self.device)
      ## ones = ones.index_select(0, gt)
      ## size.append(N)
      ## return ones.view(*size)
      ### return torch.index_select(gt_onehot, 2, indice)
      return gt_onehot

  ## Training  ================================================================
  def train(self):
      ## torch.autograd.set_detect_anomaly(False)
      ## torch.autograd.profiler.profile(False)
      ## torch.autograd.profiler.emit_nvtx(False)
      
      if os.path.isfile(self.model_to_load):
          # Load the pretrained Encoder (by gpu)
          if torch.cuda.is_available():
              checkpoint = torch.load(self.model_to_load)
          # Load model by cpu
          else:
              checkpoint = torch.load(self.model_to_load, map_location=self.device)
          ## torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(), 'optim_state_dict': self.optim.state_dict(), 'cont_optim_state_dict': self.cont_optim.state_dict(), 'scheduler': self.lr_scheduler.state_dict() }, prefix+'.pt')
          self.unet.load_state_dict(checkpoint['model_state_dict'], strict=True)
          self.epoch = checkpoint['epoch']
          self.optim.load_state_dict(checkpoint['optim_state_dict'])
          ## self.optim.zero_grad()
          if self.cont_optim is not None:
              self.cont_optim.load_state_dict(checkpoint['cont_optim_state_dict'])
              ## self.cont_optim.zero_grad()
          self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
          ## if self.adain:
          ##     self.mean = checkpoint['model_state_dict']['mean']
          ##     self.std = checkpoint['model_state_dict']['std']
          print('%s is Successfully Loaded from %s'%(self.model_type, self.model_to_load))
      
      self.unet.train(True)
      
      ## init_loss = 0.0 
      best_score = 0.0
      epoch_perf = { 'epoch_recall': [], 'epoch_precision': [], 'epoch_f1': [], 'lrs': [], 'grad_norm': [] }

      param_num = self.count_parameters(self.unet)
      print(f'Params in the model: {param_num}')
      
      ## scaler = GradScaler()

      epoch_pretrained = 0
      for epoch in range(self.epoch, self.num_epochs):
          b_time = time.time()
          unet_path = os.path.join(self.model_path, '%s-%d.pt' %(self.model_type, epoch))
          epoch_loss = { 'loss': 0.0, 'diff_loss': 0.0, 'iw_loss': 0.0, 'cont_loss': 0.0, 'contrast_loss': 0.0 }
          ## epoch_loss = 0.0
          ## epoch_iw_loss = 0.0
          ## epoch_gram_loss = 0.0
          total_norm = 0.0
          grad_not_scaled=0.0
          ls_perf = { 'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'jaccard': 0.0, 'length': 0.0, 'score': 0.0 }

          ## b_time = time.time()
          dataloader = iter(self.ls_train)
          ## a_time = time.time()
          ## print(f'dataloader spending {a_time - b_time} seconds')

          ## self.optim.zero_grad()
          ## if self.mode == 'training_content-loss':
          ##     self.cont_optim.zero_grad()
          # for i in tqdm(range(len(dataloader))):
          for i in range(len(self.ls_train)):
              ## train gradient initialization
              ## if epoch == 0 and self.grad_init:
              ##     p = torch.tensor(len(list(filter(labda p: p.grad is not None, self.unet.parameters()))))
              
              ## b_time = time.time()
              image_pair, ground_pair, filename = next(dataloader) ## b (* n) * c * h * w
              images, post_images = image_pair
              grounds, post_grounds = ground_pair

              ## if self.model_type == 'DGAttU_Net':
              ##     b, n, c, h, w = images.size()
              ##     images = images.reshape(-1, c, h, w)
              ##     grounds = grounds.reshape(-1 , c, h, w)

              ## a_time = time.time()
              ## print(f'next data spending {a_time - b_time} seconds')
              images = images.to(self.device)
              indice = torch.tensor([0]).cpu()
              GT = torch.index_select(grounds, -3, indice).to(self.device)
              if post_grounds is not None:
                  if epoch < epoch_pretrained:
                      post_images = images.detach().clone()
                      post_GT = GT.detach().clone()
                  else:
                      post_images = post_images.to(self.device)
                      post_GT = torch.index_select(post_grounds, -3, indice).to(self.device)

              ## with autocast():
              ## losses = []
              ## losses = Variable(torch.zeros(1), requires_grad=True).cuda()
              ## gram_losses = Variable(torch.zeros(1), requires_grad=True).cuda()
              ## losses = torch.zeros(1, requires_grad=True).to(self.device)
              ## iw_losses = torch.zeros(1, requires_grad=True).to(self.device)
              ## cont_losses = torch.zeros(1, requires_grad=True).to(self.device)
              iw_loss = 0.0
              cont_loss = 0.0

              ## gram_losses = torch.zeros(1, requires_grad=True).to(self.device)
              ## gram_losses = torch.tensor([0.0], requires_grad=True).to(self.device)
              
              ## if self.model_type == 'AttU_Net' or self.model_type == 'U_Net':
              ## if self.meada == True:
              ##     MEADA = ModelMEADA(self.flags, self.unet)
              ##     me_images, me_GT = MEADA.maximize(self.flags, images, GT)
              ##     images = torch.cat((images, me_images), 0)
              ##     GT = torch.cat((GT, me_GT), 0)
              ## b_time = time.time()
              ## print(f'images size: {images.size()}')
              ## print(f'GT size: {GT.size()}')
              SR, (GT, post_GT), output = self.infer((images, post_images), (GT, post_GT), state=self.mode)
              ## print(f'SR size: {SR.size()}')
              ## print(f'GT size: {GT.size()}')
              ## if post_grounds is not None:
              ##     print(f'post_GT size: {post_GT.size()}')
              ##     print(output['d1_diff'].size())
              ## print(f'output keys {output.keys()}')
              
              
              ## a_time = time.time()
              ## print(f'infer spending {a_time - b_time} seconds')
              
              ## elif self.model_type == 'STAU_Net':
              ##     SR, GT, gram_loss = self.infer(images, GT)
              ##     if not self.warmup and self.multi_style:
              ##         ## losses.append(torch.sum(loss_style)*5.0)
              ##         ## print(f'=== gram_loss {gram_loss}')
              ##         gram_losses = gram_losses + gram_loss
              ##         ## gram_losses.append(gram_loss)
              ##         ## print(f'self.gram_loss {self.gram_loss}')
              
              ## for name, param in self.unet.cont_att.named_parameters():
              ##     if param.requires_grad:
              ##         print(f'{name} requires grad')

              ## if self.loss_type == 'BCEDice':
              ## b_time = time.time()
              loss = torch.zeros(1, requires_grad=False).to(self.device)
              #### loss, BCE_loss, other_loss = self.criterion(SR, post_GT)
              ## a_time = time.time()
              ## print(f'loss spending {a_time - b_time} seconds')
              ## losses.append(loss)
              ## losses = losses + loss

              output_key = output.keys()
              if 'iw_loss' in output_key:
                  iw_loss = torch.sum(output['iw_loss'])
              if 'cont_loss' in output_key:
                  ## print(output['cont_loss'])
                  cont_loss = torch.sum(output['cont_loss'])
 
              ### print('SR.min {}, SR.max {}, loss {}'.format(SR.min(), SR.max(), loss.item()))

              ## if self.adain:
              ##     style_loss = calc_gram_style_loss(SR, GT)
              ##     epoch_loss += style_loss

              # gradient scaling ==============================
              ## clip_grad_norm_(self.unet.parameters(), max_norm=2.0, norm_type=2)
              ## p.register_hook(lambda grad: (grad + 1e-5) * 2 if grad.data.norm(2).item() < 1e-3 else grad)
              ## for p in list(filter(lambda p: p.grad is not None, self.unet.parameters())):
              ##     ## if torch.sum(torch.isnan(p)):
              ##     ##    　print(f'sum isnan {torch.sum(torch.isnan(p))}')
              ##     
              ##     ## p.register_hook(lambda g: g.clamp(-5.0, 5.0))
              ##     param_norm = p.grad.data.norm(2).item()
              ##     grad_not_scaled = grad_not_scaled + param_norm ** 2
              ##     if param_norm < 1e-2:
              ##         p.grad = 1e2 * (p.grad + 1e-5)
              ##         ## p.grad = torch.ones_like(p.grad)
              ##         param_norm = p.grad.data.norm(2).item()
              ##     total_norm = total_norm + param_norm ** 2
              ## total_norm = total_norm ** 0.5
              ## grad_not_scaled = grad_not_scaled ** 0.5
              # ===============================================
    
              # Backprop + optimize
              ## for l in losses:
              ##     if l.requires_grad == False:
              ##         print('\n**Warning: one of the loss.requires_grad is False!!\n')
            
              ## with torch.autograd.set_detect_anomaly(True):
              ## if gram_losses != 0.0:
              ##     total_loss = losses + gram_losses
              ## else:
              ##     total_loss = losses
              ## total_loss = total_loss / self.iters_to_accumulate
              ## total_loss = losses + gram_losses

              ## scaler.scale(total_loss).backward()
              if epoch < epoch_pretrained:
                  total_loss = loss + iw_loss + cont_loss
              else:
                  #### bn, c, h, w = output['d1_diff'].size()
                  b, n, c, h, w = SR.size()
                  bn = b*n
                  diff_GT = (post_GT*2 - GT).reshape(bn, 1*h*w)
                  diff_onehot = self.get_onehot(diff_GT, c).permute(0, -1, 1)   ## bn * c * hw
                  ### mseLoss = torch.nn.MSELoss()
                  ### diff_loss = mseLoss(output['d1_diff'].view(b*n, c, h*w), diff_onehot)
                  diff_loss, _, _ = BCEDiceLoss(penalty_weight=10.0)(output['d1_diff'].view(bn, c, h*w), diff_onehot.float())
                  contrast_loss = ContrastLoss()(SR, diff_onehot.view(SR.size()))
                  ### print(f'contrast_loss {contrast_loss}')
                  epoch_loss['diff_loss'] = epoch_loss['diff_loss'] + diff_loss.item()
                  epoch_loss['contrast_loss'] = epoch_loss['contrast_loss'] + contrast_loss.item()
                  ## total_loss = loss + diff_loss + iw_loss + cont_loss
                  total_loss = diff_loss + contrast_loss
              total_loss.backward()
              ## print(f'--------------- {i} loss {loss.item()} --------------')
              
              ## print(f'i: {i}')
              if (i + 1) % self.iters_to_accumulate == 0 or (i + 1) == len(self.ls_train) / self.batch_size:
                  ## self.unet.iw_optim.step()
                  ## self.unet.iw_optim.zero_grad()
                  if self.mode == 'training_content-loss':
                      self.cont_optim.step()
                      self.cont_optim.zero_grad()
                      ## print('cont_optim stepped')
                  ## print(f'{i} stepped')
                      
                  self.optim.step()
                  self.optim.zero_grad()
                  ## scaler.step(self.optim)
                  ## scaler.update()
                  ## for p in self.unet.parameters():
                  ##     p.grad = None

              epoch_loss['loss'] = epoch_loss['loss'] + loss.item()
              if 'iw_loss' in output_key:
                  epoch_loss['iw_loss'] = epoch_loss['iw_loss'] + iw_loss.item()
              if 'cont_loss' in output_key:
                  epoch_loss['cont_loss'] = epoch_loss['cont_loss'] + cont_loss.item()
              ## epoch_gram_loss = epoch_gram_loss + gram_losses.item()
              
              # End warmup
              ## if self.multi_style and self.warmup:
              ##     if epoch == 0:
              ##         init_loss = total_loss.clone().detach()
              ##     else:
              ##         recent_loss = total_loss.clone().detach()
              ##         loss_ratio = recent_loss / init_loss
              ##         if loss_ratio < 0.8:
              ##             self.warmup = False

              if self.task_type == 'CF':
                  result = torch.sigmoid(output['d1_diff'].view(bn, c, h*w).detach().cpu().float())
                  ls_perf = self.cal_perf(ls_perf, result, diff_onehot.detach().cpu())
              else:
                  result = torch.sigmoid(SR.detach().cpu().float())
                  ls_perf = self.cal_perf(ls_perf, result, post_GT.detach().cpu())


          del dataloader
          gc.collect()

          ls_perf = self.score_average(ls_perf)
          epoch_perf['grad_norm'].extend([total_norm])
        
          # Print the log info
          print('\nlr: {}, {}'.format(epoch, self.optim.param_groups[0]['lr']))
          ## print(f'Gradient Norm: {total_norm}, Gradient Not Scaled: {grad_not_scaled}')
          ## print({k: v for k, v in epoch_loss.items() if v > 0.0})
          print(epoch_loss)
          ## if 'cont_loss' in output_key:
          ##     print(f'Content Loss: {cont_loss}')
          ## if 'iw_loss' in output_key:
          ##     print(f'IW Loss: {epoch_iw_loss}')
          ## if epoch_gram_loss != 0.0:
          ##     print(f'epoch_gram_loss: {epoch_gram_loss}') 
          ##     print(f'Style mean and std: {self.mean[0].mean()}, {self.mean[1].mean()}, {self.mean[2].mean()}, {self.mean[3].mean()}, {self.std[0].mean()}, {self.std[1].mean()}, {self.std[2].mean()}, {self.std[3].mean()}')
          a_time = time.time()
          print(f'{a_time - b_time} sec/epoch')
          print('Epoch [%d/%d], Loss: %.8f \n[Training] Recall: %.4f, Precision: %.4f, F1: %.4f, Jaccard: %.4f' % (epoch+1, self.num_epochs, total_loss, ls_perf['recall'], ls_perf['precision'], ls_perf['f1'], ls_perf['jaccard']))

          epoch_perf, valid_perf, valid_set = self.valid(epoch=epoch, epoch_perf=epoch_perf)
          
          self.lr_scheduler.step()
          if self.lr_scheduler2 is not None:
              self.lr_scheduler2.step()
        
          # Save Best U-Net model
          best_score = self.save_model(self.unet, unet_path, valid_perf, valid_set, epoch, best_score)

          ## self.scheduler_warmup.step(epoch)

      ## Gradient_Norm Graph
      plt.plot(range(len(epoch_perf['grad_norm'])), epoch_perf['grad_norm'], color='red', label='Gradient Norm')
      plt.legend(loc='upper left')
      plt.xlabel('epoch')
      plt.title('Gradient Norm')
      plt.savefig(f'{self.val_root}/gradient_norm.png')
      plt.close()
      print('gradient norm graph saved.')
      
      ## Validation Performance Graph
      fig, ax1 = plt.subplots()
      ax1.set_xlabel('epoch')
      ax1.set_ylabel('Recall/Precision/F1')
      ax1.plot(range(len(epoch_perf['epoch_recall'])), epoch_perf['epoch_recall'], color='red', label='Recall')
      ax1.plot(range(len(epoch_perf['epoch_precision'])), epoch_perf['epoch_precision'], color='blue', label='Precision')
      ax1.plot(range(len(epoch_perf['epoch_f1'])), epoch_perf['epoch_f1'], color='orange', label='F1')
      ax2 = ax1.twinx()
      ax2.set_ylabel('Learning rate')##, color='green')
      ax2.plot(range(len(epoch_perf['lrs'])), epoch_perf['lrs'], 'g--', label='Learning rate')
      ax2.tick_params(axis='y')##, labelcolor='green')
      ax1.legend(loc='upper left')
      ax2.legend(loc='upper right')
      fig.tight_layout() 
      plt.title('Validation Performance')
      plt.subplots_adjust(top=0.85)
      plt.savefig(f'{self.val_root}/valid_performance.png')
      plt.close()
      print('performance graph saved.')
  
  ## Validation ================================================================
  def valid(self, epoch, epoch_perf):
    a_time = time.time()
    self.unet.train(False)
    self.unet.eval()
        
    with torch.no_grad():
        valid_perf = { 'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'jaccard': 0.0, 'length': 0.0, 'score': 0.0}
        ## valid_perf = { 'acc': 0.0, 'SE': 0.0, 'SP': 0.0, 'PC': 0.0, 'F1': 0.0, 'JS': 0.0, 'DC': 0.0, 'length': 0.0, 'score': 0.0}

        valid_set = []
        valid_loader = iter(self.ls_valid) 
        # for i in tqdm(range(len(valid_loader))):
        for i in range(len(valid_loader)):
            image_pair, ground_pair, filename = next(valid_loader) ## b * c * h * w
            images, post_images = image_pair
            grounds, post_grounds = ground_pair
            images = images.to(self.device)
            indice = torch.tensor([0]).cpu()
            GT = torch.index_select(grounds, -3, indice).to(self.device)
            if post_grounds is not None:
                post_images = post_images.to(self.device)
                post_GT = torch.index_select(post_grounds, -3, indice).to(self.device)
                
            ## if self.model_type == 'AttU_Net' or self.model_type == 'U_Net':
            ## SR, GT, _ = self.infer(images, GT, state='validation')
            SR, (GT, post_GT), output = self.infer((images, post_images), (GT, post_GT), state='validation')
            ## elif self.model_type == 'STAU_Net':
            ##     SR, GT, _ = self.infer(images, GT)

            if self.task_type == 'CF':
                bn, c, h, w = output['d1_diff'].size()
                diff_GT = (post_GT*2 - GT).reshape(bn, 1*h*w)
                diff_onehot = self.get_onehot(diff_GT, 4).permute(0, -1, 1)   ## bn * 3 * hw
                result = torch.sigmoid(output['d1_diff'].view(bn, c, h*w).detach().cpu().float())
                valid_perf = self.cal_perf(valid_perf, result, diff_onehot.detach().cpu())
            else:
                result = torch.sigmoid(SR.detach().cpu().float())
                valid_perf = self.cal_perf(valid_perf, result, GT.detach().cpu())
            ## print(f'before SR {SR.size()}, {SR[0, 0, :, :].max()}, {SR[0, 0, :, :].min()}')
            ## print(f'before GT {gt.size()}, {GT[0, 0, :, :].max()}, {GT[0, 0, :, :].min()}')
            ## print(f'after SR {result.size()}, {SR[0, 0, :, :].max()}, {SR[0, 0, :, :].min()}')
            ## print(f'after GT {GT.size()}, {GT[0, 0, :, :].max()}, {GT[0, 0, :, :].min()}')
            ## print(f'before expand {result.size()}, {GT.size()}')
            ## img = images.reshape(-1, 3, self.image_size, self.image_size) / 255.0
            img = images.detach().cpu()/255.0
            ## SR_expand = torch.squeeze((result > self.threshold).float(), 1).expand(-1, 3, -1, -1)
            ## GT_expand = torch.squeeze(GT.detach().cpu(), 1).expand(-1, 3, -1, -1)

            if self.task_type == 'CF':
                result = result.reshape(bn, c, h, w)
                GT = GT.reshape(bn, -1, h, w)
                post_GT = post_GT.reshape(bn, -1, h, w)
                c_num = 4
                step = 255.0/c_num
                SR_expand = torch.zeros((bn, 3, h, w)).cpu().float()
                for id in range(1, c_num):
                    indice = torch.tensor([id]).cpu()
                    result_c = torch.index_select(result, -3, indice)
                    SR_expand = SR_expand + (result_c*id*step).expand(-1, 3, -1, -1)
                GT_expand = GT.detach().cpu().expand(-1, 3, -1, -1)
                postGT_expand = post_GT.detach().cpu().expand(-1, 3, -1, -1)
                valid_set.append([img, GT_expand, postGT_expand, SR_expand])

            else:
                if len(result.size()) == 5:
                    b, n, c, h, w = result.size()
                    result = result.reshape(b*n, c, h, w)
                    GT = GT.reshape(b*n, c, h, w)
                SR_expand = (result > self.threshold).float().expand(-1, 3, -1, -1)
                GT_expand = GT.detach().cpu().expand(-1, 3, -1, -1)
            ## print(f'save sized {img.size()}, {GT_expand.size()}, {SR_expand.size()}')
            ## print(f'saved type {type(img)}, {type(GT_expand)}, {type(SR_expand)}')
            ## print(f'value {img.min()}, {img.max()}, {SR_expand.max()}, {SR_expand.min()}')
                valid_set.append([img, GT_expand, SR_expand])

        valid_perf = self.score_average(valid_perf)
        # Print the log info
        print(f'{time.time() - a_time} sec/epoch')
        print('[Validation] Recall: %.4f, Precision: %.4f, F1: %.4f, Jaccard: %.4f' % (valid_perf['recall'], valid_perf['precision'], valid_perf['f1'], valid_perf['jaccard']))
        ## print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (valid_perf['acc'], valid_perf['SE'], valid_perf['SP'], valid_perf['PC'], valid_perf['F1'], valid_perf['JS'], valid_perf['DC']))
        ## if not self.warmup and self.multi_style:
        ##     print('\n=================== Warmup ends ===================\n')

        epoch_perf['epoch_recall'].extend([valid_perf['recall']])
        epoch_perf['epoch_precision'].extend([valid_perf['precision']])
        epoch_perf['epoch_f1'].extend([valid_perf['f1']])
        epoch_perf['lrs'].extend([self.optim.param_groups[0]['lr']])

            
        del valid_loader
        gc.collect()

        return epoch_perf, valid_perf, valid_set
        

  ## Testing
  def test(self): 
    # Load the pretrained Encoder (by gpu)
    print(f'cuda available {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        checkpoint = torch.load(self.model_to_load)
    # Load model by cpu
    else:
        checkpoint = torch.load(self.model_to_load, map_location=self.device)
    ## print(checkpoint)
    
    ### load .pkl model file
    ## self.unet.load_state_dict(checkpoint['model_state_dict'], strict=False)
    ## self.optim.load_state_dict(checkpoint['optim_state_dict'])
    ## self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
    ## if self.adain:
    ##     self.mean = checkpoint['model_state_dict']['mean']
    ##     self.std = checkpoint['model_state_dict']['std']
    ## =================================================
    ### load .pt model file
    self.unet.load_state_dict(checkpoint['model_state_dict'], strict=True)
    ## self.optim.load_state_dict(checkpoint['optim_state_dict'])
    ## self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
    ## if self.adain:
    ##     self.mean = checkpoint['model_state_dict']['module.mean']
    ##     self.std = checkpoint['model_state_dict']['module.std']
    ## =================================================
    pre_epoch = checkpoint['epoch']
    print('%s is Successfully Loaded from %s'%(self.model_type, self.model_to_load))

    ### network info.
    ## for n, p in self.unet.state_dict().items():
    ##     print(n, p.size(), p)
    param_num = self.count_parameters(self.unet)
    print(f'Params in the model: {param_num} \n')

    self.unet.train(False)
    self.unet.eval()
    
    for test_n in range(len(self.inf_path)):
        self.inf_folder = self.inf_path[test_n]
        print(f'test and infer folder: {self.test_path[test_n]}, {self.inf_folder}')
        try:
            os.makedirs(self.inf_folder)
        except FileExistsError:
            pass
        try:
            os.makedirs(self.inf_folder+'overlap/')
        except FileExistsError:
            pass
        try:
            os.makedirs(self.inf_folder+'Mask/')
        except FileExistsError:
            pass
        with torch.no_grad():
            test_perf = { 'recall': 0.0, 'precision': 0.0, 'f1': 0.0, 'jaccard': 0.0, 'length': 0.0, 'score': 0.0}
            ## test_perf = { 'acc': 0.0, 'SE': 0.0, 'SP': 0.0, 'PC': 0.0, 'F1': 0.0, 'JS': 0.0, 'DC': 0.0, 'length': 0.0, 'score': 0.0}
            dataloader = iter(self.test_loader[test_n])
            for i in range(len(dataloader)):
                if self.model_type == 'DamageNet':
                    (images, post_images), (grounds, post_grounds), filename = next(dataloader)
                    images = images.to(self.device)
                    indice = torch.tensor([0]).to(grounds.device)
                    GT = torch.index_select(grounds, -3, indice).to(self.device)
                    if post_grounds is not None:
                        post_images = post_images.to(self.device)
                        post_GT = torch.index_select(post_grounds, -3, indice).to(self.device)

                    SR, (GT, post_GT), output = self.infer((images, post_images), (GT, post_GT), state='testing')

                else:
                    (images, _), (grounds, _), filename = next(dataloader)
                    images = images.to(self.device)
                    ## print(f'test size {images.size()}, {grounds.size()}')
                    indice = torch.tensor([0]).cpu()
                    gt = torch.index_select(grounds, -3, indice).to(self.device)
                    ## if self.model_type == 'STAU_Net':
                    SR, GT, _ = self.infer((images, None), gt, state='testing')
                    ## else:
                    ##     SR, GT = self.infer(images, gt, 'testing')

                result = torch.sigmoid(SR.detach().cpu().float())
                test_perf = self.cal_perf(test_perf, result, GT.detach().cpu())
                
                ## SR_expand = torch.squeeze((result > self.threshold).float(), 1).expand(-1, 3, -1, -1)
                ## GT_expand = torch.squeeze(GT.detach().cpu(), 1).expand(-1, 3, -1, -1)
                if len(result.size()) == 5:
                    b, n, c, h, w = result.size()
                    result = result.reshape(b, c, h, w)
                    GT = GT.reshape(b, c, h, w)
                    images = images.reshape(b, 3, h, w)
                SR_expand = (result > self.threshold).float().expand(-1, 3, -1, -1)
                GT_expand = GT.detach().cpu().expand(-1, 3, -1, -1)
                ## print(f'expand size check {SR_expand.size()}, {GT_expand.size()}')
                img = images.detach().cpu() / 255.0
                for n in range(result.size()[0]):
                    res = []
                    res.append([img[n], GT_expand[n], SR_expand[n]])
                    ## res_overlap = torch.cat((GT[n].detach().cpu(), result[n]), dim=0)
                    save_image(res[0], self.inf_folder + filename[n] + '.png', nrow=len(res[0]))
                    save_image(SR_expand[n], self.inf_folder + 'Mask/' + filename[n] + '.tif', nrow=1)
                    ## save_image(res_overlap, self.inf_folder + 'overlap/' + filename[n] + '.tif')
            test_perf = self.score_average(test_perf)
            print('[Testing] Recall: %.4f, Precision: %.4f, F1: %.4f, Jaccard: %.4f' % (test_perf['recall'], test_perf['precision'], test_perf['f1'], test_perf['jaccard']))
            ## print('[Testing] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (test_perf['acc'], test_perf['SE'], test_perf['SP'], test_perf['PC'], test_perf['F1'], test_perf['JS'], test_perf['DC']))
