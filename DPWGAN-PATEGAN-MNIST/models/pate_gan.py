import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
import math
from utils.architectures import Generator, Discriminator, GeneratorCNN, DiscriminatorCNN
from utils.helper import weights_init, pate, moments_acc
import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PATE_GAN:
    def __init__(self, input_dim, z_dim, num_teachers, target_epsilon, target_delta, conditional=True, epochs=100):
        # self.generator = Generator(z_dim, input_dim, conditional).cuda().double()
        self.generator = GeneratorCNN().to(device)
        # self.student_disc = Discriminator(input_dim, wasserstein=False).cuda().double()
        self.student_disc = DiscriminatorCNN().to(device)
        # self.teacher_disc = [Discriminator(input_dim, wasserstein=False).cuda().double() for _ in range(num_teachers)]
        self.teacher_disc = [DiscriminatorCNN().to(device) for _ in range(num_teachers)]
        self.generator.apply(weights_init)
        self.student_disc.apply(weights_init)
        # self.z_dim = z_dim
        self.z_dim = 100
        self.num_teachers = num_teachers
        # self.num_teachers = 1
        for i in range(num_teachers):
            self.teacher_disc[i].apply(weights_init)

        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.conditional = conditional
        self.epochs = epochs

    def train(self, x_train, y_train, hyperparams):
        batch_size = hyperparams.batch_size
        num_teacher_iters = hyperparams.num_teacher_iters
        # num_teacher_iters = 10
        num_student_iters = hyperparams.num_student_iters
        # num_student_iters = 10
        num_moments = hyperparams.num_moments
        lap_scale = hyperparams.lap_scale
        class_ratios = None
        if self.conditional:
            class_ratios = torch.from_numpy(hyperparams.class_ratios)

        # real_label = 1
        # fake_label = 0

        # alpha = torch.cuda.DoubleTensor([0.0 for _ in range(num_moments)])
        alpha = torch.FloatTensor([0.0 for _ in range(num_moments)]).to(device)
        # l_list = 1 + torch.cuda.DoubleTensor(range(num_moments))
        l_list = 1 + torch.FloatTensor(range(num_moments)).to(device)

        criterion = nn.BCELoss().to(device)
        adversarial_loss = torch.nn.MSELoss().to(device)

        # optimizer_g = optim.Adam(self.generator.parameters(), lr=hyperparams.lr)
        # optimizer_g = optim.Adam(self.generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizer_g = optim.Adam(self.generator.parameters(), lr=5e-5, betas=(0.5, 0.999))
        # optimizer_sd = optim.Adam(self.student_disc.parameters(), lr=hyperparams.lr)
        # optimizer_sd = optim.Adam(self.student_disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
        optimizer_sd = optim.Adam(self.student_disc.parameters(), lr=5e-5, betas=(0.5, 0.999))
        # optimizer_td = [optim.Adam(self.teacher_disc[i].parameters(), lr=hyperparams.lr) for i in range(self.num_teachers)]
        # optimizer_td = [optim.Adam(self.teacher_disc[i].parameters(), lr=2e-4, betas=(0.5, 0.999)) for i in range(self.num_teachers)]
        optimizer_td = [optim.Adam(self.teacher_disc[i].parameters(), lr=5e-5, betas=(0.5, 0.999)) for i in range(self.num_teachers)]
        # tensor_data = data_utils.TensorDataset(torch.cuda.DoubleTensor(x_train), torch.cuda.DoubleTensor(y_train))
        tensor_data = data_utils.TensorDataset(torch.FloatTensor(x_train).to(device), torch.LongTensor(y_train).to(device))

        train_loader = []
        for teacher_id in range(self.num_teachers):
            # print(len(tensor_data), self.num_teachers) 324, 10
            start_id = teacher_id * len(tensor_data) / self.num_teachers
            end_id = (teacher_id + 1) * len(tensor_data) / self.num_teachers if teacher_id != (
                    self.num_teachers - 1) else len(tensor_data)
            # print(start_id, end_id)
            train_loader.append(data_utils.DataLoader(torch.utils.data.Subset( \
                tensor_data, range(int(start_id), int(end_id))), batch_size=batch_size, shuffle=True))

        # steps = 0
        epsilon = 0
        epoch = 0
        while epsilon < self.target_epsilon:
            if epoch >= self.epochs:
                break

            # train the teacher discriminators
            for t_2 in range(num_teacher_iters):
                for i in range(self.num_teachers):
                    inputs, categories = None, None
                    for b, data in enumerate(train_loader[i], 0):
                        inputs, categories = data
                        break

                    # modify
                    inputs, categories = inputs.to(device), categories.to(device)
                    inputs = inputs.view(-1, 1, 28, 28)  # for CNN
                    valid = torch.FloatTensor(inputs.shape[0], 1).fill_(1.0).to(device)
                    fakelabel = torch.FloatTensor(inputs.shape[0], 1).fill_(0.0).to(device)

                    # (1) for teacher discriminator: train with real
                    optimizer_td[i].zero_grad()
                    # label = torch.full((inputs.size()[0],), 1.0).to(device)
                    # output = self.teacher_disc[i].forward(torch.cat([inputs, categories.unsqueeze(1).double()], dim=1))
                    output = self.teacher_disc[i].forward(inputs, categories.view(-1).long())
                    # err_d_real = criterion(output, label.double())
                    # err_d_real = adversarial_loss(output, valid)
                    err_d_real = - torch.log(torch.sigmoid(output)).sum()
                    # err_d_real.backward() # difference !!!!!!

                    # (1) for teacher discriminator: train with fake
                    z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).to(device) # 64
                    # z = torch.randn(batch_size, self.z_dim).to(device)

                    if self.conditional:
                        # category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).unsqueeze(1).cuda().double()  # 32
                        category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).view(-1).to(device).long()  # 32
                        # category = torch.from_numpy(np.random.randint(0, 2, inputs.size()[0])).long().to(device)
                        # a = torch.cat([z.double(), category], dim=1)
                        # print(a.shape)
                        # fake = self.generator(torch.cat([z.double(), category], dim=1))
                        fake = self.generator(z, category)
                        output = self.teacher_disc[i].forward(fake.detach(), category)
                    else:
                        fake = self.generator(z.double())
                        output = self.teacher_disc[i].forward(fake)

                    # err_d_fake = criterion(output, label.double())
                    # err_d_fake = adversarial_loss(output, fakelabel)
                    err_d_fake = - torch.log(1-torch.sigmoid(output)).sum()
                    # print('err_d_real', err_d_real)
                    # print('err_d_fake', err_d_fake)
                    d_loss_teacher = (err_d_real + err_d_fake) / 2
                    d_loss_teacher.backward()
                    # err_d_fake.backward()
                    optimizer_td[i].step()

            # (2) train the student discriminator
            for t_3 in range(num_student_iters):
                z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).cuda()
                # z = torch.randn(batch_size, self.z_dim).to(device)

                if self.conditional:
                    # category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).unsqueeze(1).cuda().double()
                    category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).view(-1).to(device).long()
                    # category = torch.from_numpy(np.random.randint(0, 2, inputs.size()[0])).long().to(device)
                    # fake = self.generator(torch.cat([z.double(), category], dim=1))
                    fake = self.generator(z, category)
                    # adding noises !!!!!!!!!
                    # predictions, clean_votes = pate(torch.cat([fake.detach().view(fake.shape[0], -1), category.float().view(-1, 1)], dim=1), self.teacher_disc, lap_scale)
                    # predictions, clean_votes = pate(fake.detach().view(fake.shape[0], -1), category.long(), self.teacher_disc, lap_scale)
                    results = torch.Tensor(len(self.teacher_disc), fake.size()[0]).type(torch.int64)
                    for i in range(len(self.teacher_disc)):
                        output = self.teacher_disc[i].forward(fake.detach().view(fake.shape[0], -1), category)
                        output = torch.sigmoid(output) # linear to probability
                        pred = (output > 0.5).type(torch.Tensor).squeeze()
                        results[i] = pred
                    # print('result', results)
                    clean_votes = torch.sum(results, dim=0).unsqueeze(1).to(device)
                    # print('clear votes', clean_votes)
                    noise = torch.from_numpy(np.random.laplace(loc=0, scale=1 / lap_scale, size=clean_votes.size())).to(device)
                    noisy_results = clean_votes + noise
                    predictions = (noisy_results > len(self.teacher_disc) / 2).to(device)
                    # print('prediction: ',predictions)

                    outputs = self.student_disc.forward(fake.detach(), category)
                else:
                    fake = self.generator(z.double())
                    predictions, clean_votes = pate(fake.detach(), self.teacher_disc, lap_scale) # add noise: line 15 in algorithm
                    outputs = self.student_disc.forward(fake.detach())

                # update the moments: line 16
                alpha = alpha + moments_acc(self.num_teachers, clean_votes, lap_scale, l_list)

                # update student: line 20
                optimizer_sd.zero_grad()
                outputs = torch.sigmoid(outputs)
                err_sd = criterion(outputs, predictions.float().to(device)) # only bce
                # err_sd = predictions.float() * torch.log(outputs) + (1-predictions.float()) * torch.log(1-outputs)
                # err_sd = predictions.float() * torch.log(outputs) + (1-predictions.float()) * torch.log(1-outputs)
                # err_sd = err_sd.sum()
                # print('err_sd', err_sd)
                err_sd.backward()
                optimizer_sd.step()

            # (3) generator: train the generator
            optimizer_g.zero_grad()
            z = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).to(device)
            # z = torch.randn(batch_size, self.z_dim).to(device)
            # label = torch.full((inputs.size()[0],), real_label).cuda()
            valid = torch.FloatTensor(batch_size, 1).fill_(1.0).to(device)

            if self.conditional:
                # category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).unsqueeze(1).cuda().double()
                category = torch.multinomial(class_ratios,  inputs.size()[0], replacement=True).view(-1).to(device).long()
                # category = torch.from_numpy(np.random.randint(0, 2, inputs.size()[0])).long().to(device)
                # fake = self.generator(torch.cat([z.double(), category], dim=1))
                fake = self.generator(z, category)
                output = self.student_disc(fake, category.long())
            else:
                fake = self.generator(z.double())
                output = self.student_disc.forward(fake)

            # err_g = criterion(output, label.double())
            # err_g = adversarial_loss(output, valid)
            output = torch.sigmoid(output)
            err_g = torch.log(1-output).sum() # log loss
            # print('err_g', err_g)
            err_g.backward()
            optimizer_g.step()

            # Calculate the current privacy cost
            epsilon = min((alpha - math.log(self.target_delta)) / l_list)
            if epoch % 100 == 0:
                # save image
                from torchvision.utils import save_image
                import os
                noise = torch.randn(16, 100).to(device)
                cat = torch.cat([torch.zeros(8).long(), torch.ones(8).long()], dim=0).to(device)
                synthetic = self.generator(noise, cat)  # 4*64, for CNN
                synthetic = synthetic.reshape(16, 1, 28, 28)
                path = './data/lapscale' + str(lap_scale)
                if not os.path.exists('./data/lapscale' + str(lap_scale)):
                    os.mkdir('./data/lapscale' + str(lap_scale))
                visual_path = path + '/' + str(epoch) + '.png'
                save_image(synthetic, visual_path, nrow=4, normalize=False)

                # print
                print("Epoch : ", epoch,  "/", self.epochs, "Loss SD : ", err_sd.item(), "Loss G : ", err_g.item(), "Epsilon : ",
                      epsilon.item(), time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

            epoch += 1

    def generate(self, num_rows, class_ratios, batch_size=1000):
        steps = num_rows // batch_size
        synthetic_data = []
        if self.conditional:
            class_ratios = torch.from_numpy(class_ratios)
        for step in range(steps):
            # noise = torch.randn(batch_size, self.z_dim).to(device)
            noise = torch.Tensor(batch_size, self.z_dim).uniform_(0, 1).to(device)
            if self.conditional:
                # cat = torch.multinomial(class_ratios, batch_size, replacement=True).unsqueeze(1).cuda().double()
                cat = torch.multinomial(class_ratios, batch_size, replacement=True).view(-1).to(device).long()
                # cat = torch.from_numpy(np.random.randint(0, 2, batch_size)).long().to(device)
                # synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = self.generator(noise, cat)
                synthetic = torch.cat([synthetic.reshape(batch_size, -1), cat.float().reshape(batch_size, 1)], dim=1)

            else:
                synthetic = self.generator(noise)

            synthetic_data.append(synthetic.cpu().data.numpy())

        if steps * batch_size < num_rows:
            size = num_rows - steps * batch_size
            # noise = torch.randn(num_rows - steps * batch_size, self.z_dim).to(device)
            noise = torch.Tensor(size, self.z_dim).uniform_(0, 1).to(device)
            if self.conditional:
                # cat = torch.multinomial(class_ratios, num_rows - steps * batch_size, replacement=True).unsqueeze(1).cuda().double()
                cat = torch.multinomial(class_ratios, num_rows - steps * batch_size, replacement=True).view(-1).to(device).long()
                # cat = torch.from_numpy(np.random.randint(0, 2, num_rows - steps * batch_size)).long().to(device)
                # synthetic = self.generator(torch.cat([noise.double(), cat], dim=1))
                synthetic = self.generator(noise, cat)
                # synthetic = torch.cat([synthetic, cat], dim=1)
                synthetic = torch.cat([synthetic.reshape(size, -1), cat.float().reshape(size, 1)], dim=1)
            else:
                synthetic = self.generator(noise)
            synthetic_data.append(synthetic.cpu().data.numpy())

        return np.concatenate(synthetic_data)