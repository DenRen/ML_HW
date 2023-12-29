import torch
from rare_traffic_sign_solution import CustomNetwork, FeaturesLoss
from sys import argv, exit

if __name__ == "__main__":
    if len(argv) != 2:
        print(f"Enter: {argv[0]} <path>")
        exit()
    
    path = argv[1]
    if False:
        module = CustomNetwork.load_from_checkpoint(path, map_location="cpu")
        torch.save(module.state_dict(), "simple_model.pth")
    else:
        module = CustomNetwork.load_from_checkpoint(path, map_location="cpu", 
                                                    features_criterion=FeaturesLoss(2.0),
                                                    internal_features=1024,
                                                    pretrained=False)
        torch.save(module.state_dict(), "improved_features_model.pth")
