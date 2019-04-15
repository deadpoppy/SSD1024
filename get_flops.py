import torch
from flops_benchmark import add_flops_counting_methods


def get_flops(net, input_size=(300, 300)):
    input_size = (1, 3, input_size[0], input_size[1])
    input = torch.randn(input_size)
    input = torch.autograd.Variable(input.cuda())

    net = add_flops_counting_methods(net)
    net = net.cuda().eval()
    net.start_flops_count()

    _ = net(input)

    return net.compute_average_flops_cost()/1e9/2


# example
if __name__ == '__main__':

    from ssd.modeling.vgg_ssd import build_ssd_model
    from ssd.config import cfg
    '''
    '''

    cfg.merge_from_file("configs/ssd512_voc0712.yaml")

    cfg.freeze()
    model = build_ssd_model(cfg)
    input_size = (1024, 1024)
    #ssd_net = model.eval()
    ssd_net = model.cuda()


    total_flops = get_flops(ssd_net, input_size)

    # For default vgg16 model, this shoud output 31.386288 G FLOPS
    print("The Model's Total FLOPS is : {:.6f} G FLOPS".format(total_flops))
