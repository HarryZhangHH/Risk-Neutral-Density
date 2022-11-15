import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from algorithm import RiskNeutralDensityFunctions
from visualization.plotting import relu, single_line_plot, multiple_lines_plot, multi_axs_line_plot
pd.options.mode.chained_assignment = None  # default='warn'

# --------------------------------------------------------------------------- #
# Parse command line arguments (CLAs):
# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Playground', prog='RISK NEUTRAL DENSITY')
parser.add_argument('--rule', default='trapezoidal', type=str, 
                    choices=['trapezoidal', 'simpson', 'irresimpson'], help='numerical method (choice: trapezoidal, simpson, irregular simpson)')
parser.add_argument('--polynomial', default='hermite', type=str, 
                    choices=['hermite', 'laguerre', 'legendre'], help='polynomials (choice: hermite, laguerre, legendre)')
parser.add_argument('--transformation', default='normal', type=str, 
                    choices=['normal', 'log'], help='whether want to trasform using h_k(x) (choice: normal, log)')
parser.add_argument('--k', default=100, type=int, help='truncation term, k>1, also = N_end')
parser.add_argument('--stock_l', default=3500, type=float, help='stock price lowest boundary')
parser.add_argument('--stock_h', default=4900, type=float, help='stock price highest boundary')
parser.add_argument('--stock_i', default=5, type=float, help='stock price interval')
parser.add_argument('--a', default=5100, type=float, help='a in normal transformed h_k(x)')
parser.add_argument('--b', default=1800, type=float, help='b in normal transformed h_k(x)')
parser.add_argument('--line_num', default=7, type=int, help='how many lines do you want to show in the Estimated RND result plot?')
# parser.add_argument('--error', default='True', type=str, help='whether want to plot the difference between the baseline?')
parser.add_argument('--N_num', default=100, type=int, help='how many N do you want to show in the expansion coefficient plot?')
parser.add_argument('--N_start', default=70, type=int, help='The initial N to plot')
parser.add_argument('--activation', default=None, type=str, 
                    choices=[None, 'relu', 'abs'], help='The activation for estimate and error (choice: relu, abs, None)')
parser.add_argument('--k_idx', nargs="+", default=[], type=int, help='the specfic line you want to plot')
parser.add_argument('--k_range', nargs="+", default=[], type=int, help='the k range you want to plot')
# parser.print_help()
# --------------------------------------------------------------------------- #

def process_data():
    # load data
    data = pd.read_csv('2021-04-20.csv')
    data.date = pd.to_datetime(data['date'], infer_datetime_format=True, format='%d/%m/%Y')
    data.exdate = pd.to_datetime(data['exdate'], infer_datetime_format=True, format='%d/%m/%Y')
    date_time_str = '21/05/2021'
    # date_time_str = '16/07/2021'
    # date_time_str = '28/04/2021'
    date_time_obj = datetime.strptime(date_time_str, '%d/%m/%Y')
    data_processed = data[data.exdate == date_time_obj]
    data_processed.sort_values(by='strike_price',inplace=True)
    data_processed.reset_index(drop=True, inplace=True)
    return data_processed

def calculate_trading_day(date1, date2):
    # generating dates
    dates = (date1 + timedelta(idx + 1)
            for idx in range((date2 - date1).days))
    # summing all weekdays
    return sum(1 for day in dates if day.weekday() < 5)

def main():
    global args
    args = parser.parse_args()

    # real data don't need 'S0' and 'sigma'
    S0, sigma = None, None
    data = process_data()
    print(f"The numerical method is {args.rule} rule\nThe polynomial function is {args.polynomial} function\n\
        The hyperparameters are {args}\nLET'S START!!!")
    
    config = {
        'polynomial': args.polynomial,
        'rule': args.rule,
        'transformation': args.transformation,
        'k': args.k,
        'T': data.time2exp[0],
        'rate': data.Rf[0],
        'y': np.arange(args.stock_l,args.stock_h,args.stock_i),
        'K': data.strike_price,
        'F0': data.forward[0],
        'S0': S0,
        'sigma': sigma,
        'a': args.a,
        'b': args.b
    }
    rnd = RiskNeutralDensityFunctions(config, True)
    rnd.BS = data.mid_price
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
            line_num = np.abs(rnd.k-args.N_start)
        else:
            line_num = args.line_num
        for idx in range(args.N_start, rnd.k, int((rnd.k-args.N_start)/line_num)):
            Idx.append(idx)
    density, f_k_mean = rnd.RND_f(Idx)
    density = np.array(density).reshape((len(Idx),-1))
    error = np.diff(density,axis=0)

    if args.activation == 'relu':
        density = relu(density)
    elif args.activation == 'abs':
        density = np.abs(density)

    fname = '5-21' + args.polynomial[:3] + '_' + args.rule[:3] + '_' +  args.transformation

    x = rnd.y
    Idx1 = list(map(lambda x: 'N = ' + str(x), Idx))
    xlabel = 'Stock Price'
    if args.transformation == 'normal':
        # x = np.log(rnd.y)
        x = rnd.y
        y = density.T
        # y = density.T/x.reshape((-1,1))
        # xlabel = '(Log) Stock Price'
        xlabel = 'Stock Price'
        # print(len(density.T))
        # print(len(rnd.y))
        # density.T/rnd.y
    multiple_lines_plot(x=x, y=y, labels=Idx1, title='RN Density:Exact vs Estimation', xlabel=xlabel, ylabel='Density', fname=fname+'_density')
    
    Idx2 = ['Diff of '+str(Idx[x])+' and '+str(Idx[x+1]) for x in range(len(Idx)-1)]
    multiple_lines_plot(x=x, y=np.log10(np.abs(error)).T, labels=Idx2, title='(Log) Estimation Error', xlabel=xlabel, ylabel='Log Absolute Error', fname=fname+'_error')

    i = 4  # interval
    if args.N_num < args.k:
        single_line_plot(x=np.array([i for i in range(args.N_num)])[:args.N_num:i], y=np.log(np.abs(f_k_mean))[:args.N_num:i], title='Expansion Coefficients', 
            xlabel='N', ylabel='Abs Coeff (with log)', fname=fname+'_expansion')
        # single_line_plot(x=np.array([i for i in range(args.N_num)]), y=np.abs(f_k_mean)[:args.N_num], title='Expansion Coefficients', 
        #     xlabel='N', ylabel='Abs Coeff (without log)', fname=fname+'_expansion2')
    else:
        single_line_plot(x=np.arange(f_k_mean.shape[0])[:args.N_num:i], y=np.log(np.abs(f_k_mean))[:args.N_num:i], title='Expansion Coefficients', 
            xlabel='N', ylabel='Abs Coeff (with log)', fname=fname+'_expansion')
        # single_line_plot(x=np.arange(f_k_mean.shape[0]), y=np.abs(f_k_mean), title='Expansion Coefficients', 
        #     xlabel='N', ylabel='Abs Coeff (without log)', fname=fname+'_expansion2')

if __name__ == '__main__':
    main()