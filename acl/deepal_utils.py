from torchvision import transforms
from avalanche.models.resnet18_64 import ResNet18_64
from avalanche.models import ResNet18
from avalanche.models import SimpleMLP
from avalanche.query_strategies import *
from avalanche.training import *

def get_net(name):
    if name == 'MNIST':
        return SimpleMLP(num_classes=10)
    if name == 'CIFAR10':
        return ResNet18(num_classes=10)
    elif name == 'CIFAR100':
        return ResNet18(num_classes=100)
    elif name == 'TinyImageNet':
        return ResNet18_64(num_classes=200)
    else:
        raise NotImplementedError
        
def get_cl_strategy(model, optimizer, criterion, plugin, eval_plugin, device, img_size, args):
    if args.cl_strategy == 'ER':
        return Replay(
            model, optimizer, criterion,
            plugins=plugin,
            evaluator=eval_plugin,
            train_mb_size=args.batch_size,
            eval_mb_size=args.batch_size,
            device=device,
            train_epochs=args.epochs,
            mem_size=args.mem_size,
        )
    elif args.cl_strategy == 'GSS':
        return GSS_greedy(
            model, optimizer, criterion,
            plugins=plugin,
            evaluator=eval_plugin,
            train_mb_size=args.batch_size,
            eval_mb_size=args.batch_size,
            device=device,
            train_epochs=args.epochs,
            mem_size=args.mem_size,
            input_size=img_size,
        )
    elif args.cl_strategy == 'DER':
        return DER(
            model, optimizer, criterion,
            plugins=plugin,
            evaluator=eval_plugin,
            train_mb_size=args.batch_size,
            eval_mb_size=args.batch_size,
            device=device,
            train_epochs=args.epochs,
            mem_size=args.mem_size,
        )
    elif args.cl_strategy == 'ER_ACE':
        return ER_ACE(
            model, optimizer, criterion,
            plugins=plugin,
            evaluator=eval_plugin,
            train_mb_size=args.batch_size,
            eval_mb_size=args.batch_size,
            batch_size_mem=args.batch_size,
            device=device,
            train_epochs=args.epochs,
            mem_size=args.mem_size,
        )
    else:
        raise NotImplementedError
