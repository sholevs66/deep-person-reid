import torchreid
import argparse
import torch


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_name', type=str, help='Name of arhcitecture', default='repvgg_a0')
    parser.add_argument('--weights', type=str, help='path to weights')
    parser.add_argument('--output_path', type=str, help='path for saved onnx', default='model.onnx')
    parser.add_argument('--num_classes', type=int, help='#classes model was trained on', default=1000)
    args = parser.parse_args()

    model = torchreid.models.build_model(str(args.model_name), int(args.num_classes))

    torchreid.utils.load_pretrained_weights(model, str(args.weights))
    model.eval()

    # create dummy input
    imgs = torch.zeros((1, 3, 256, 128))

    # check if repvgg model -> deploy
    if hasattr(model, 'deploy'):
        from torchreid.models.repvgg import repvgg_model_convert
        model = repvgg_model_convert(model)
        features = model(imgs)

    torch.onnx.export(model, imgs, str(args.output_path), input_names=['test_input'], output_names=['test_output'])
    print(features.shape)


if __name__ == '__main__':
    main()
