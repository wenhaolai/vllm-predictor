"""用法：python tests/inspect_hidden_state.py /path/to/hidden_state.pt"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description="读取保存的 hidden state 信息")
    parser.add_argument("path", help="保存的 .pt 文件路径")
    args = parser.parse_args()

    data = torch.load(args.path, map_location="cpu", weights_only=True)
    tensor = data["hidden_state"]
    print("request_id:", data.get("request_id"))
    print("num_prompt_tokens:", data.get("num_prompt_tokens"))
    print("token_position:", data.get("token_position"))
    print("shape:", tuple(tensor.shape))
    print("dtype:", tensor.dtype)


if __name__ == "__main__":
    main()
