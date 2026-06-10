import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

from PIL import Image 
from torchvision import transforms 

img = Image.open("images.jpeg").convert("RGB")
to_tensor = transforms.ToTensor()
img_tensor = to_tensor(img).unsqueeze(0)

# resizing the image tensors 

def resize_tensor(img,scale):
    _,_,h,w = img.shape

    new_h = int(h*scale) 
    new_w = int(w*scale)
    
    return F.interpolate(img, size = (new_h,new_w),mode='bicubic',align=False)

#scaling the images 
scales = [1,0.9,0.8,0.7]

pyramid = []

for s in scales : 
    pyramid.append(resize_tensor(img_tensor,s))



pairs = []

for target in pyramid: 
    img = F.interpolate(target,scale_factor = 0.5 , mode='bicubic',align_corners=False)
    pairs.append((inp,target))

# making ythe CNN 

class zssr(nn.Module):
    def __init__(self):
        super().__init__(self):

        self.conv1 = nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(64,64,3,kernel_size=3,padding=1)

    def forward(self,x):
        residual = x 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.conv4(x)

        return x + residual 
    
device = "cuda" if torch.cuda.is_available() else "cpu" 
model = zssr().to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(),lr = 1e-3)



epochs = 500 
for epoch in range(epochs): 
    total_loss = 0 
    for lr_img ,hr_img in pairs:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        lr_up = F.interpolate(lr_img , size = hr_img.shape[-2:],mode='bicubic',align_corners=False)
        pred = model(lr_up)
        loss = criterion(pred,hr_img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()


    if epoch%50==0 : 

        print(f"Epoch{epoch}Loss{total_loss:.5f}")





    sr_input = F.interpolate(img_tensor.to(device),scale_factor=2,mode='bicubic',align_corners = False)

    with torch.no_grad():

        sr_output = model(sr_input)


sr_output = sr_output.squeeze(0)
sr_output = sr_output.clamp(0,1)
to_pil = transforms.ToPILImage()
result = to_pil(sr_output.cpu())

result.save("SR_result.png")