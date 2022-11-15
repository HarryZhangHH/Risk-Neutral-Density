import argparse
import numpy as np
from algorithm import RiskNeutralDensityFunctions
from visualization.plotting import relu, single_line_plot, multiple_lines_plot, multi_axs_line_plot

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Playground', prog='RISK NEUTRAL DENSITY')
parser.add_argument('--rule', default='trapezoidal', type=str, 
                    choices=['trapezoidal', 'simpson', 'irresimpson'], help='numerical method (choice: trapezoidal, simpson, irregular simpson)')
parser.add_argument('--polynomial', default='hermite', type=str, 
                    choices=['hermite', 'laguerre', 'legendre'], help='polynomials (choice: hermite, laguerre, legendre)')
parser.add_argument('--k', default=200, type=int, help='truncation term, k>1, also = N_end')
parser.add_argument('--S0', default=5, type=int, help='current stock price')
parser.add_argument('--F0', default=5, type=int, help='forward price')
parser.add_argument('--sigma', default=0.1, type=float, help='volatility, 0<sigma<1')
parser.add_argument('--trading_days', default=252, type=int, help='trading days')
parser.add_argument('--time_to_maturity', default=30, type=int, help='time to maturity')
parser.add_argument('--rate', default=0, type=float, help='rate')
parser.add_argument('--strike_l', default=1, type=float, help='strike price lowest boundary')
parser.add_argument('--strike_h', default=9.05, type=float, help='strike price highest boundary')
parser.add_argument('--strike_i', default=0.05, type=float, help='strike price interval')
parser.add_argument('--stock_l', default=1, type=float, help='stock price lowest boundary')
parser.add_argument('--stock_h', default=9.05, type=float, help='stock price highest boundary')
parser.add_argument('--stock_i', default=0.05, type=float, help='stock price interval')
parser.add_argument('--line_num', default=7, type=int, help='how many lines do you want to show in the Estimated RND result plot?')
# parser.add_argument('--error', default='True', type=str, help='whether want to plot the difference between the baseline?')
parser.add_argument('--N_start', default=70, type=int, help='The initial N to plot')
parser.add_argument('--N_num', default=None, type=int, help='how many N do you want to show in the expansion coefficient plot?')
parser.add_argument('--activation', default=None, type=str, 
                    choices=['relu', 'abs', None], help='The activation for estimate and error (choice: relu, abs, None)')
parser.add_argument('--transformation', default=None, type=str, 
                    choices=['normal', 'log', None], help='whether want to trasform using h_k(x) (choice: normal, log, None)')
parser.add_argument('--a', default=0, type=float, help='a in normal transformed h_k(x)')
parser.add_argument('--b', default=1, type=float, help='b in normal transformed h_k(x)')
parser.add_argument('--k_idx', nargs="+", default=[], type=int, help='the specfic line you want to plot')
parser.add_argument('--k_range', nargs="+", default=[], type=int, help='the k range you want to plot')
# parser.print_help()
# --------------------------------------------------------------------------- #

def main():
    global args
    args = parser.parse_args()

    print(f"The numerical method is {args.rule} rule\nThe polynomial function is {args.polynomial} function\n\
        The hyperparameters are {args}\nLET'S START!!!")

    config = {
        'polynomial': args.polynomial,
        'rule': args.rule,
        'transformation': args.transformation,
        'k': args.k,
        'T': args.time_to_maturity/args.trading_days,
        'rate': args.rate,
        'y': np.arange(args.stock_l,args.stock_h,args.stock_i),
        'K': np.arange(args.strike_l,args.strike_h,args.strike_i),
        'F0': args.F0,
        'S0': args.S0,
        'sigma': args.sigma,
        'a': args.a,
        'b': args.b
    }
    rnd = RiskNeutralDensityFunctions(config)
    Idx = []
    if args.k_idx:
        Idx = list(set([l if 0<l<=rnd.k else rnd.k for l in args.k_idx]))
        Idx.sort()
    elif args.k_range:
        Idx = list(set([l if 0<l<=rnd.k else rnd.k for l in args.k_range]))
        Idx.sort()
        Idx = [l for l in range(Idx[0], Idx[-1]+1)]
    else:
        if (rnd.k-args.N_start) < args.line_num:
            line_num = rnd.k-args.N_start
        else:
            line_num = args.line_num
        for idx in range(args.N_start, rnd.k, int((rnd.k-args.N_start)/line_num)):
            Idx.append(idx)
    density, f_k_mean = rnd.RND_f(Idx)
    density = np.array(density).reshape((len(Idx),-1))

    error = rnd.pdf.reshape((1,-1))-density
    # find best error
    print('====================================================')
    best_idx, min_error, index = rnd.find_best_error(density, Idx)
    print(f'Best k is {best_idx}    MSE is {min_error}')
    print('====================================================')

    if args.activation == 'relu':
        density = relu(density)
    elif args.activation == 'abs':
        density = np.abs(density)
    
    x = rnd.y
    y = np.vstack((density,rnd.pdf)).T
    Idx = list(map(lambda x: 'N = ' + str(x), Idx))
    Idx.append('Exact RND')
    xlabel = 'Stock Price'
    if args.transformation == 'log':
        x = np.log(rnd.y)
        xlabel = '(Log) Stock Price'
    
    fname = args.polynomial[:3] + '_' + args.rule[:3] + '_' + str(args.sigma) + args.transformation + str(args.time_to_maturity) 
        
    multiple_lines_plot(x=x, y=y, labels=Idx, title='RN Density:Exact vs Estimation', xlabel=xlabel, ylabel='Density', fname=fname+'_density')
    Idx.pop(-1)
    multiple_lines_plot(x=x, y=np.log10(np.abs(error)).T, labels=Idx, title='(Log) Estimation Error', xlabel=xlabel, ylabel='Log Absolute Error', fname=fname+'_error')
    
    single_line_plot(x=x, y=error[index], label=f'N = {best_idx}', title='Estimation Error', xlabel=xlabel, ylabel='Error', fname=fname+'_best')

    i = 2   # slice step interval
    if args.N_num is not None and args.N_num < args.k:
        # single_line_plot(x=np.array([i for i in range(args.N_num)])[:args.N_num:i], y=np.log10(np.abs(f_k_mean))[:args.N_num:i], title='Expansion Coefficients', 
        #     xlabel='N', ylabel='Abs Coeff (with log)', fname=fname+'_expansion')
        single_line_plot(x=np.array([i for i in range(args.N_num)]), y=np.abs(f_k_mean)[:args.N_num], title='Expansion Coefficients', 
            xlabel='N', ylabel='Abs Coeff (without log)', fname=fname+'_expansion2')
    else:
        # single_line_plot(x=np.arange(f_k_mean.shape[0])[:args.k:i], y=np.log10(np.abs(f_k_mean))[:args.k:i], title='Expansion Coefficients', 
        #     xlabel='N', ylabel='Abs Coeff (with log)', fname=fname+'_expansion')
        single_line_plot(x=np.arange(f_k_mean.shape[0]), y=np.abs(f_k_mean), title='Expansion Coefficients', 
            xlabel='N', ylabel='Abs Coeff (without log)', fname=fname+'_expansion2')
    
    

if __name__ == '__main__':
    main()