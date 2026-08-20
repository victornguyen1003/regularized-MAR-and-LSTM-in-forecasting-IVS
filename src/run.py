from preprocess import main as preprocess
from visualize_ivs import main as visualize_ivs
from run_var import main as run_var
from run_mar import main as run_mar
from run_regularized_mar import main as run_regularized_mar


def main():
    preprocess()
    visualize_ivs()
    run_var()
    run_mar()
    run_regularized_mar()


if __name__ == "__main__":
    main()