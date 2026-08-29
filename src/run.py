from preprocess import main as preprocess
from visualize_ivs import main as visualize_ivs
from run_var import main as run_var
from run_mar import main as run_mar
from finetune_regularized_mar import main as finetune_regularized_mar
from run_regularized_mar import main as run_regularized_mar
from examine_mar import main as examine_mar
from finetune_lstm import main as finetune_lstm
from run_lstm import main as run_lstm
from visualize_res import main as visualize_res


def main():
    preprocess()
    visualize_ivs()
    run_var()
    run_mar()
    finetune_regularized_mar()
    run_regularized_mar()
    examine_mar()
    finetune_lstm()
    run_lstm()
    visualize_res()



if __name__ == "__main__":
    main()