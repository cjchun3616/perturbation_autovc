import os
import time
import torch
import datetime
import torch.nn.functional as F
from models.model_vc import Generator

class Solver(object):

    def __init__(self, vcc_loader, config):
        # Data loader.
        self.vcc_loader = vcc_loader 
        
        # Model configurations.
        self.lambda_cd = config.lambda_cd
        self.dim_emb = config.dim_emb
        self.dim_pre = config.dim_pre
        self.freq = config.freq
        
        # model checkpoint and save path.
        self.resume = config.resume
        self.save_dir = config.save_dir
        
        # Training configrations.
        self.batch_size = config.batch_size
        self.num_iters = config.num_iters
        
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
        self.log_step = config.log_step
        
        # build the model.
        self.build_model()
        
    def build_model(self):
        
        self.G = Generator(self.dim_emb, self.dim_pre, self.freq)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), 1e-4)
        self.G.to(self.device)
        
        if self.resume is not None:
            print("checkpoint load at ", self.resume)
            g_checkpoint = torch.load(self.resume, map_location='cuda:0')
            self.G.load_state_dict(g_checkpoint['model_state_dict'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict']) 

    def reset_grad(self):
        # Reset the gradient.
        self.g_optimizer.zero_grad()
                      
    def train(self):
        # Set the loader.
        data_loader = self.vcc_loader
        
        # Print logs in specified order.
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']

        print('Start training...')
        start_time = time.time()
        self.G = self.G.train()
        
        for i in range(self.num_iters):
            '''Preprocess input data.'''
            data_iter = iter(data_loader)
            x_real, uttr, emb_org = next(data_iter)
                   
            x_real = x_real.to(self.device) # Clean data.
            uttr = uttr.to(self.device) # Perturbed data.
            emb_org = emb_org.to(self.device) # Speaker information.
            
            energy = torch.mean(x_real, dim=1, keepdim=True) # Extract energy.
            emb_de = emb_org.unsqueeze(-1).expand(-1, -1, energy.shape[-1]) 
            emb_de = torch.cat((emb_de, energy), dim=1)

            x_identic, x_identic_psnt, code_real = self.G(uttr, emb_de)
            
            x_identic = x_identic.squeeze()
            x_identic_psnt = x_identic_psnt.squeeze()
            
            # Calculate reconstruction loss.
            g_loss_id = F.mse_loss(x_real, x_identic)   
            g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt) 
            
            # Calculate cotent loss.
            code_reconst = self.G(x_identic_psnt, None)
            g_loss_cd = F.l1_loss(code_real, code_reconst)

            # Backward and optimize.
            g_loss = 2*(g_loss_id + g_loss_id_psnt) + self.lambda_cd * g_loss_cd # sigma=2, mu=1.
            self.reset_grad()
            g_loss.backward()
            self.g_optimizer.step()
            
            # Print out training losses.
            if i % 200 == 0: 
                print("loss_id: ", g_loss_id.item(), "loss_id_psnt: ", g_loss_id_psnt.item(), "g_loss_cd: ", g_loss_cd.item())
                
            if i % 100000 == 0:
                self.lambda_cd = self.lambda_cd * 0.9
            
            # Logging.
            loss = {}
            loss['G/loss_id'] = g_loss_id.item()
            loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
            loss['G/loss_cd'] = g_loss_cd.item()
            
            # Print out training information.
            if (i+1) % self.log_step == 0:
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
                # Save the model at checkpoint.
                torch.save({'model_state_dict': self.G.state_dict(),
                            'optimizer_state_dict':self.g_optimizer.state_dict()},
                            os.path.join(self.save_dir, f'{self.pt_name}_{i//10000+1}.pt'))
                for tag in keys:
                    log += ", {}: {:.4f}".format(tag, loss[tag])
                print(log)
                

    
    

    