from math import fabs
from runner import Runner
from arguments import get_args
from utils import make_env
if __name__ == '__main__':
    # get the params
    args = get_args()
    args.evaluate = 1
    env, args = make_env(args)
    runner = Runner(args, env)
    evaluate=args.evaluate
    if evaluate:
        for _ in range(1):
            runner.evaluate()
    else:
        runner.run()
