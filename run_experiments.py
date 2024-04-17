import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1" #! specify gpu here
from main import Main

if __name__ == '__main__':
    # run all experiments

    # classification
    main = Main(["--mode", "train",
                "--model", "classify",
                "--early_stopping", "True",
                "--freeze_backbone", "False",
                "--train_epochs", "600"])

    # main = Main(["--mode", "eval",
    #             "--model", "classify",
    #             "--checkpoint_path", "results/2023-12-20__14-56_classify_WORKING_2/lightning_logs/version_0/checkpoints/epoch=544-step=544.ckpt"])

    # # SIFT eval
    # Main(["--mode", "train",
    #     "--train_epochs", "1",
    #     "--cutoff", "0.01",
    #     "--model", "sift"])

    # # superglue eval
    # Main(["--mode", "eval",
    #     "--cutoff", "0.8",
    #     "--model", "superglue",
    #     "--superglue_model", "/home/sruiz/projects/reconcycle/superglue_training/output/train/2023-11-18_superglue_model/weights/best.pt",
    #     "--visualise", "False"])

    # # pairwise_classifier train
    # main = Main(["--mode", "train",
    #              "--backbone", "clip",
    #             "--model", "pairwise_classifier",
    #             "--freeze_backbone", "True"])
    
    # results_path = main.args.results_path

    # # pairwise_classifier eval
    # Main(["--mode", "eval",
    #     "--model", "pairwise_classifier",
    #     "--results_path", results_path])

    #! available models:
    # superglue/sift
    # pw_concat_bce/pw_cos/pw_cos2
    # triplet

    # main = Main(["--mode", "train",
    #              "--model", "pairwise_classifier",
    #              "--backbone", "superglue",
    #              "--weight_decay", "1e-4",
    #              "--freeze_backbone", "True",
    #              "--early_stopping", "False",
    #              "--train_epochs", "600",
    #              "--visualise", "False"])

    # pairwise_classifier2 train
    # doesn't use cutoff
    # main = Main(["--mode", "train",
    #              "--model", "pairwise_classifier2",
    #              "--weight_decay", "1e-4",
    #              "--freeze_backbone", "True",
    #              "--early_stopping", "False",
    #              "--train_epochs", "600",
    #              "--visualise", "False"])
    
    # pairwise_classifier3 train
    # main = Main(["--mode", "train",
    #              "--model", "pairwise_classifier3",
    #              "--cutoff", "0.9",
    #              "--freeze_backbone", "True",
    #              "--visualise", "True"])

    # cosine eval
    # Main(["--mode", "eval",
    #     "--cutoff", "0.9",
    #     "--model", "cosine",
    #     "--visualise", "True"])

    # Clip eval
    # Main(["--mode", "eval",
    #     "--cutoff", "0.9",
    #     "--model", "clip",
    #     "--visualise", "True"])
    

    # triplet
    # main = Main(["--mode", "train", 
    #              "--model", "triplet",
    #              "--cutoff", "1.0",
    #              "--weight_decay", "1e-4",
    #              "--freeze_backbone", "True",
    #              "--early_stopping", "False",
    #              "--train_epochs", "200",
    #              "--visualise", "True"])


    #! I can't seem to replicate? Did I get the cutoff wrong?
    # main = Main(["--mode", "eval", 
    #              "--model", "triplet",
    #              "--cutoff", "1.636", # learned
    #              "--weight_decay", "1e-4",
    #              "--freeze_backbone", "True",
    #              "--early_stopping", "False",
    #              "--train_epochs", "200",
    #              "--visualise", "True",
    #              "--results_path", "experiments/results/2023-04-17__16-45-14_triplet",
    #              "--checkpoint_path", "lightning_logs/version_0/checkpoints/epoch=126-step=126.ckpt"])
    