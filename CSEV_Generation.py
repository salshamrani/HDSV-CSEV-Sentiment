import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict

yelp_with_embeddings = load_from_disk("Datasets/yelp_with_embeddings")

def average_embeddings(dataset):
    """
    Compute average embeddings for positive and negative samples.
    """
    embeddings = np.array(dataset['embeddings'])
    labels = np.array(dataset['label'])
    
    positive_avg = np.mean(embeddings[labels == 1], axis=0)
    negative_avg = np.mean(embeddings[labels == 0], axis=0)
    
    return positive_avg, negative_avg

def create_averaged_dataset(original_dataset):
    """
    Create a dataset with averaged embeddings as centroids for each class.
    """
    train_pos_avg, train_neg_avg = average_embeddings(original_dataset['train'])
    val_pos_avg, val_neg_avg = average_embeddings(original_dataset['validation'])
    
    new_train = Dataset.from_dict({
        'label': [1, 0],
        'embeddings': [train_pos_avg.tolist(), train_neg_avg.tolist()]
    })
    
    new_val = Dataset.from_dict({
        'label': [1, 0],
        'embeddings': [val_pos_avg.tolist(), val_neg_avg.tolist()]
    })
    
    new_dataset = DatasetDict({
        'train': new_train,
        'validation': new_val,
        'test': original_dataset['test']
    })
    
    return new_dataset

def main():
    new_yelp = create_averaged_dataset(yelp_with_embeddings)
    
    print("New Yelp Dataset with Averaged Embeddings:")
    for split in new_yelp.keys():
        print(f"  {split}:")
        print(f"    Number of samples: {len(new_yelp[split])}")
        if split != 'test':
            print(f"    Positive samples: 1")
            print(f"    Negative samples: 1")
        else:
            labels = new_yelp[split]['label']
            print(f"    Positive samples: {sum(labels)}")
            print(f"    Negative samples: {len(labels) - sum(labels)}")
    
    new_yelp.save_to_disk("Datasets/yelp_CSEV")

if __name__ == "__main__":
    main()
