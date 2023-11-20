import torch
from mingpt.model import GPT
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.trainer import Trainer
from mingpt.utils import set_seed
import pickle
import numpy as np
import matplotlib.pyplot as plt


class SortDataset(Dataset):
    """ 
    Dataset for the Sort problem. E.g. for problem length 6:
    Input: 0 0 2 1 0 1 -> Output: 0 0 0 1 1 2
    Which will feed into the transformer concatenated as:
    input:  0 0 2 1 0 1 0 0 0 1 1
    output: I I I I I 0 0 0 1 1 2
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
    
    def __len__(self):
        return 10000 # ...
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length * 2 - 1

    def __getitem__(self, idx):
        
        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time let's try to boost the number of examples that 
            # have a large number of repeats, as this is what the model seems to struggle
            # with later in training, and they are kind of rate
            if torch.rand(1).item() < 0.5:
                if inp.unique().nelement() > self.length // 2:
                    # too many unqiue digits, re-sample
                    continue
            # figure out if this generated example is train or test based on its hash
            h = hash(pickle.dumps(inp.tolist()))
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break # ok
        
        # solve the task: i.e. sort
        sol = torch.sort(inp)[0]

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length-1] = -1
        return x, y



def train_model(seed, train_dataset, model_config, train_config):
    set_seed(seed)
    model = GPT(model_config)
    trainer = Trainer(train_config, model, train_dataset)
    trainer.run()
    return model.state_dict()

def linear_interpolation(model_weights_1, model_weights_2, alpha):
    interpolated_weights = {}
    for key in model_weights_1.keys():
        interpolated_weights[key] = alpha * model_weights_1[key] + (1 - alpha) * model_weights_2[key]
    return interpolated_weights

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

def linear_interpolation(model_weights_1, model_weights_2, alpha):
    interpolated_weights = {}
    for key in model_weights_1.keys():
        interpolated_weights[key] = alpha * model_weights_1[key] + (1 - alpha) * model_weights_2[key]
    return interpolated_weights



def main():

    # print an example instance of the dataset
    set_seed(3407)
    train_dataset = SortDataset('train')
    test_dataset = SortDataset('test')
    gpt_type = 'gpt2'

    # Initialize model configuration
    model_config = GPT.get_default_config()
    model_config.model_type = gpt_type
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()

    # Initialize model and move it to the selected device (GPU if available)
    model = GPT(model_config)

    # # Initialize dataset
    # train_dataset = YourDataset()  # Ensure that your dataset is compatible with GPU usage

    # Trainer configuration
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    train_config.max_iters = 2000
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_dataset)

    # Initialize trainer
    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()


    #train a new model
    # Initialize model configuration
    model_config = GPT.get_default_config()
    model_config.model_type = gpt_type
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
     # Initialize model and move it to the selected device (GPU if available)
    model1 = GPT(model_config)
     # Trainer configuration
    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    train_config.max_iters = 2000
    train_config.num_workers = 0
    trainer1 = Trainer(train_config, model1, train_dataset)

    # Initialize trainer
    trainer1.set_callback('on_batch_end', batch_end_callback)
    trainer1.run()


    weights_model_1 = model.state_dict()
    weights_model_2 = model1.state_dict()


    # # Choose a range for alpha, e.g., 0 to 1 in steps of 0.1
    # for alpha in np.linspace(0, 1, 11):
    #     # Interpolate between the two models
    #     interpolated_weights = linear_interpolation(weights_model_1, weights_model_2, alpha)

    #     # Load the interpolated weights into a new model instance
    #     interpolated_model = GPT(model_config)
    #     interpolated_model.load_state_dict(interpolated_weights)

    #     # # Evaluate the interpolated model
    #     # metric = evaluate_model(interpolated_model, test_dataset)
    #     # print(f'Alpha: {alpha}, Evaluation Metric: {metric}')

    def eval_split(trainer, split, max_batches):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        n = train_dataset.length # naugy direct access shrug
        results = []
        mistakes_printed_already = 0
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            y = y.to(trainer.device)
            # isolate the input pattern alone
            inp = x[:, :n]
            sol = y[:, -n:]
            # let the model sample the rest of the sequence
            cat = model.generate(inp, n, do_sample=False) # using greedy argmax, not sampling
            sol_candidate = cat[:, n:] # isolate the filled in sequence
            # compare the predicted sequence to the true sequence
            correct = (sol == sol_candidate).all(1).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 3: # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    print("GPT claims that %s sorted is %s but gt is %s" % (inp[i].tolist(), sol_candidate[i].tolist(), sol[i].tolist()))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return ((len(results)-rt.sum())/len(results))*100
    
    with torch.no_grad():
        test_score  = eval_split(trainer, 'test',  max_batches=50)
        test_score1  = eval_split(trainer1, 'test',  max_batches=50)

    
    alphas = np.linspace(0, 1, 100)
    train_errors = []
    test_errors = []

    for alpha in alphas:
        interpolated_weights = linear_interpolation(weights_model_1, weights_model_2, alpha)
        model_config = GPT.get_default_config()
        model_config.model_type = gpt_type
        model_config.vocab_size = train_dataset.get_vocab_size()
        model_config.block_size = train_dataset.get_block_size()
        interpolated_model = GPT(model_config)
        interpolated_model.load_state_dict(interpolated_weights)

        accuracy = eval_split(trainer, 'test',  max_batches=50)
        test_errors.append(accuracy)
        accuracy = eval_split(trainer, 'train',  max_batches=50)
        train_errors.append(accuracy)

        print(f'Alpha: {alpha}, Accuracy: {accuracy}')
    
    E_sup_train = max(train_errors)
    E_sup_test = max(test_errors)
    mean_test = (test_score+test_score1)/2

    print("E_sup_test:", E_sup_test)
    print("mean_test:", mean_test)

    error_barrier_height = E_sup_test - mean_test
    print("error_barrier_height:", error_barrier_height)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(alphas, train_errors, marker='o', label='Train Error')
    plt.plot(alphas, test_errors, marker='x', label='Test Error')
    plt.xlabel('Alpha')
    plt.ylabel('Error Rate')
    plt.title('Training and Testing Error')
    plt.ylim(0, 20)  # Adjust the limits of the y-axis as necessary
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
