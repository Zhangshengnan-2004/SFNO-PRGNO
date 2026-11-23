"""Optional absolute-epoch LR schedule for resumed CAVE training."""
import math


def add_arguments(parser):
    parser.add_argument('--resume-lr', type=float, default=None,
                        help='Explicit LR after optimizer restoration; requires --resume')
    parser.add_argument('--lr-decay-epochs', type=int, nargs='*', default=[],
                        help='Completed epoch counts after which LR is multiplied by gamma')
    parser.add_argument('--lr-gamma', type=float, default=0.5)


def validate(args):
    if args.resume_lr is not None and (not args.resume or not math.isfinite(args.resume_lr) or args.resume_lr <= 0):
        raise ValueError('--resume-lr requires --resume and a finite positive value')
    if not math.isfinite(args.lr_gamma) or not 0 < args.lr_gamma <= 1:
        raise ValueError('--lr-gamma must be in (0,1]')
    if any(e <= 0 for e in args.lr_decay_epochs) or len(set(args.lr_decay_epochs)) != len(args.lr_decay_epochs):
        raise ValueError('LR decay epochs must be distinct positive integers')
    if args.lr_decay_epochs and args.resume_lr is None:
        raise ValueError('Explicit --resume-lr is required for continuation decay')


def apply(optimizer, args, completed_epochs):
    if args.resume_lr is None:
        return optimizer.param_groups[0]['lr']
    lr = args.resume_lr * args.lr_gamma ** sum(completed_epochs >= e for e in args.lr_decay_epochs)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr
