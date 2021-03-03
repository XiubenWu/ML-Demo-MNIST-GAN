# 机器学习demo
* 采用改进的CGAN模型
* 在MNIST数据集上进行处理
* Datas中存放了**MNIST**的数据文件
* images存放了训练过程中产生的效果图像(每1000个batchsize生成一次图像)
```python
"MNISTDemo.py"

def main():
    mode = 0
    if mode == 0:
        cgan_demo()
    elif mode == 1:
        cgan_new_demo()
    else:
        print("mode error")
```
直接修改mode的值可调用不同的模型

```python
def cgan_demo():
    generator = Module.GeneratorCGAN(opt.latent_dim, opt.n_classes, img_shape) 
    discriminator = Module.DiscriminatorCGAN(opt.n_classes, img_shape)

    Function.train_cgan(generator, discriminator, data_loader, opt.n_epochs, opt.lr, opt.b1, opt.b2,
                        opt.latent_dim, opt.n_classes, cuda, fist_train=True)

def cgan_new_demo():
    show_data_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "Datas/",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=100,
        shuffle=True,
    )
    generator = Module.GeneratorCGANNew(img_shape)
    discriminator = Module.DiscriminatorCGANNew(img_shape)

    Function.train_cgan_new(generator, discriminator, data_loader, show_data_loader, opt.n_epochs, opt.lr, opt.b1,
                            opt.b2,
                            cuda, first_train=True)

```
最后一个参数first_train更改为False调用已有模型(**确保参数路径存在对应模型，否则使用True从头训练**)

**Note:** 该项目为机器学习备份。