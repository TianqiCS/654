import torch
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
from utils.rdp_accountant import compute_rdp, get_privacy_spent
from utils.architectures import Generator, Discriminator, GeneratorCNN, DiscriminatorCNN
from utils.helper import weights_init
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DP_WGAN:
    def __init__(self, input_dim, z_dim, target_epsilon, target_delta, conditional=True, epochs=100):
        self.input_dim = input_dim
        self.z_dim = 100 # fixed !!!!!!!!!!
        # self.generator = Generator(z_dim, input_dim, conditional).cuda().double()
        self.generator = GeneratorCNN().to(device)
        # self.discriminator = Discriminator(input_dim, wasserstein=True).cuda().double()
        self.discriminator = DiscriminatorCNN().to(device)
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.target_epsilon = target_epsilon # !!!
        self.target_delta = target_delta # !!! fix 0.001
        self.conditional = conditional
        self.epochs = epochs

    def train(self, x_train, y_train, hyperparams, private=False):
        batch_size = hyperparams.batch_size
        micro_batch_size = hyperparams.micro_batch_size
        lr = hyperparams.lr
        clamp_upper = hyperparams.clamp_upper
        clamp_lower = hyperparams.clamp_lower
        clip_coeff = hyperparams.clip_coeff # WGAN
        sigma = hyperparams.sigma # ???
        class_ratios = None

        if self.conditional:
            class_ratios = torch.from_numpy(hyperparams.class_ratios)

        if torch.cuda.is_available():
            data_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.cuda.FloatTensor(x_train), torch.cuda.LongTensor(y_train)),
                                            batch_size=batch_size, shuffle=True)
        else:
            data_loader = data_utils.DataLoader(data_utils.TensorDataset(torch.FloatTensor(x_train), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)

        # optimizer_g = optim.RMSprop(self.generator.parameters(), lr=lr)
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        # optimizer_d = optim.RMSprop(self.discriminator.parameters(), lr=lr)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        adversarial_loss = torch.nn.MSELoss().to(device)

        # one = torch.DoubleTensor([1]).to(device)
        one = torch.Tensor([1.0]).to(device)
        mone = one * -1
        epsilon = 0
        gen_iters = 0
        steps = 0
        epoch = 0



        while epsilon < self.target_epsilon:
            if epoch >= self.epochs:
                break

            data_iter = iter(data_loader)
            i = 0

            while i < len(data_loader):

                # Update Critic

                for p in self.discriminator.parameters():
                    p.requires_grad = True

                if gen_iters < 25 or gen_iters % 500 == 0:
                    disc_iters = 100

                else:
                    disc_iters = 5

                j = 0
                while j < disc_iters and i < len(data_loader):
                    j += 1

                    # clamp parameters to a cube

                    # for p in self.discriminator.parameters():
                    #     p.data.clamp_(clamp_lower, clamp_upper)

                    data = data_iter.next()
                    i += 1

                    # train with real
                    optimizer_d.zero_grad()


                    inputs, categories = data
                    inputs, categories = inputs.to(device), categories.to(device)
                    inputs = inputs.view(-1, 1, 28, 28) # for CNN
                    valid = torch.FloatTensor(inputs.shape[0], 1).fill_(1.0).to(device)
                    fakelabel = torch.FloatTensor(inputs.shape[0], 1).fill_(0.0).to(device)

                    # err_d_real = self.discriminator(torch.cat([inputs, categories.unsqueeze(1).double()], dim=1))
                    err_d_real = self.discriminator(inputs, categories.view(-1).long()) # For CNN
                    d_real_loss = adversarial_loss(err_d_real, valid)


                    #################### core code: clippping and add noise on gradients of discrimimator
                    if private:
                        # For privacy, clip the avg gradient of each micro-batch

                        clipped_grads = {
                            name: torch.zeros_like(param) for name, param in self.discriminator.named_parameters()}

                        for k in range(int(err_d_real.size(0) / micro_batch_size)):
                            err_micro = err_d_real[k * micro_batch_size: (k + 1) * micro_batch_size].mean(0).view(1)
                            err_micro.backward(one, retain_graph=True)
                            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), clip_coeff) # clip: line 9
                            for name, param in self.discriminator.named_parameters():
                                clipped_grads[name] += param.grad
                            self.discriminator.zero_grad()

                        for name, param in self.discriminator.named_parameters():
                            # add noise here, ####### line 7 in algorithm 1
                            param.grad = (clipped_grads[name] + torch.Tensor(
                                clipped_grads[name].size()).normal_(0, sigma * clip_coeff).to(device)) / (
                                                     err_d_real.size(0) / micro_batch_size)

                        steps += 1

                    else:
                        err_d_real.mean(0).view(1).backward(one)

                    # train with fake
                    noise = torch.randn(batch_size, self.z_dim).to(device)
                    if self.conditional:
                        # category = torch.multinomial(class_ratios,  batch_size, replacement=True).unsqueeze(1).to(device).double()
                        category = torch.multinomial(class_ratios,  batch_size, replacement=True).view(-1).to(device).long() # for CNN
                        # category =  torch.from_numpy(np.random.randint(0, 2, batch_size)).long().to(device)
                        # fake = self.generator(torch.cat([noise.double(), category], dim=1))
                        fake = self.generator(noise, category) # for CNN
                        # err_d_fake = self.discriminator(torch.cat([fake.detach(), category], dim=1)).mean(0).view(1)
                        err_d_fake = self.discriminator(fake.detach(), category).mean(0).view(1) # for CNN

                    else:
                        fake = self.generator(noise)
                        err_d_fake = self.discriminator(fake.detach()).mean(0).view(1)

                    # err_d_fake.backward(mone)
                    d_fake_loss = adversarial_loss(err_d_fake, fakelabel)
                    d_loss = (d_real_loss + d_fake_loss) / 2
                    d_loss.backward()
                    optimizer_d.step() # update the discriminator

                # Update Generator: no clipping
                for p in self.discriminator.parameters():
                    p.requires_grad = False

                optimizer_g.zero_grad()
                noise = torch.randn(batch_size, self.z_dim).to(device)
                if self.conditional:
                    # category = torch.multinomial(class_ratios,  batch_size, replacement=True).unsqueeze(1).to(device).double()
                    category = torch.multinomial(class_ratios,  batch_size, replacement=True).view(-1).to(device).long() # for CNN
                    # category = torch.from_numpy(np.random.randint(0, 2, batch_size)).long().to(device)
                    # fake = self.generator(torch.cat([noise.double(), category], dim=1))
                    fake = self.generator(noise, category) # for CNN
                    # err_g = self.discriminator(torch.cat([fake, category.double()], dim=1)).mean(0).view(1)
                    err_g = self.discriminator(fake, category.long()).mean(0).view(1) # for CNN
                else:
                    fake = self.generator(noise)
                    err_g = self.discriminator(fake).mean(0).view(1)

                # err_g.backward(one)
                g_loss = adversarial_loss(err_g, valid)
                g_loss.backward()

                optimizer_g.step() # update the generator
                gen_iters += 1

            epoch += 1
            if private:
                # Calculate the current privacy cost using the accountant
                max_lmbd = 4095
                lmbds = range(2, max_lmbd + 1)
                rdp = compute_rdp(batch_size / x_train.shape[0], sigma, steps, lmbds)
                epsilon, _, _ = get_privacy_spent(lmbds, rdp, target_delta=1e-5) # compute the epsilon: utils/rdp_accountant.py
            else:
                if epoch > hyperparams.num_epochs:
                    epsilon = np.inf

            SAVE = True
            if SAVE and epoch % 1 == 0:
                from torchvision.utils import save_image
                import os
                noise = torch.randn(16, 100).to(device)
                # cat = torch.cat([torch.zeros(8, 1).double(), torch.ones(8, 1).double()], dim=0).to(device)
                cat = torch.cat([torch.zeros(8).long(), torch.ones(8).long()], dim=0).to(device)
                # cat = torch.multinomial(class_ratios, batch_size, replacement=True).unsqueeze(1).cuda().double()
                # synthetic = self.generator(torch.cat([noise.double(), cat], dim=1)) # 4*64
                synthetic = self.generator(noise, cat) # 4*64, for CNN
                synthetic = synthetic.reshape(16, 1, 28, 28)
                path = './data/sigma'+str(sigma)
                if not os.path.exists('./data/sigma'+str(sigma)):
                    os.mkdir('./data/sigma'+str(sigma))
                visual_path = path + '/'+str(epoch)+'.png'
                save_image(synthetic, visual_path, nrow=4, normalize=False)


            print("Epoch :", epoch, "/", self.epochs, "Loss D real : ", err_d_real.mean(0).view(1).item(),
                  "Loss D fake : ", err_d_fake.item(), "Loss G : ", err_g.item(), "Epsilon spent : ", epsilon, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))





    def generate(self, num_rows, class_ratios, batch_size=1000):
        steps = num_rows // batch_size
        synthetic_data = []
        if self.conditional:
            class_ratios = torch.from_numpy(class_ratios)
        for step in range(steps):
            noise = torch.randn(batch_size, self.z_dim).to(device)
            if self.conditional:
                # cat = torch.multinomial(class_ratios,  batch_size, replacement=True).unsqueeze(1).to(device)
                cat = torch.multinomial(class_ratios,  batch_size, replacement=True).view(-1).to(device).long()
                # synthetic = self.generator(torch.cat([noise, cat], dim=1))
                synthetic = self.generator(noise, cat)
                synthetic = torch.cat([synthetic.reshape(batch_size, -1), cat.float().reshape(batch_size, 1)], dim=1)

            else:
                synthetic = self.generator(noise)

            synthetic_data.append(synthetic.cpu().data.numpy())

        if steps*batch_size < num_rows:
            size = num_rows - steps*batch_size
            noise = torch.randn(num_rows - steps*batch_size, self.z_dim).to(device)

            if self.conditional:
                # cat = torch.multinomial(class_ratios, num_rows - steps*batch_size, replacement=True).unsqueeze(1).to(device).double()
                cat = torch.multinomial(class_ratios, num_rows - steps*batch_size, replacement=True).view(-1).to(device).long()
                synthetic = self.generator(noise, cat)
                synthetic = torch.cat([synthetic.reshape(size, -1), cat.float().reshape(size, 1)], dim=1)
            else:
                synthetic = self.generator(noise)
            synthetic_data.append(synthetic.cpu().data.numpy())

        return np.concatenate(synthetic_data)
