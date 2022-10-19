import torch
import paddle
paddle.set_device("cpu")

num_layers = 12
old = torch.load("./init_weights/ldmbert.pt", map_location="cpu")
new = {}
new["embeddings.word_embeddings.weight"] = old["transformer.token_emb.weight"].numpy()
new["embeddings.position_embeddings.weight"] = old["transformer.pos_emb.emb.weight"].numpy()
for i in range(num_layers):
    double_i = 2 * i
    double_i_plus1 = 2 * i + 1
    # convert norm
    new[f"encoder.layers.{i}.norm1.weight"] = old[f"transformer.attn_layers.layers.{double_i}.0.weight"].numpy()
    new[f"encoder.layers.{i}.norm1.bias"] = old[f"transformer.attn_layers.layers.{double_i}.0.bias"].numpy()
    
    new[f"encoder.layers.{i}.self_attn.q_proj.weight"] = old[f"transformer.attn_layers.layers.{double_i}.1.to_q.weight"].t().numpy()
    new[f"encoder.layers.{i}.self_attn.k_proj.weight"] = old[f"transformer.attn_layers.layers.{double_i}.1.to_k.weight"].t().numpy()
    new[f"encoder.layers.{i}.self_attn.v_proj.weight"] = old[f"transformer.attn_layers.layers.{double_i}.1.to_v.weight"].t().numpy()
    new[f"encoder.layers.{i}.self_attn.out_proj.weight"] = old[f"transformer.attn_layers.layers.{double_i}.1.to_out.weight"].t().numpy()
    new[f"encoder.layers.{i}.self_attn.out_proj.bias"] = old[f"transformer.attn_layers.layers.{double_i}.1.to_out.bias"].numpy()

    new[f"encoder.layers.{i}.norm2.weight"] = old[f"transformer.attn_layers.layers.{double_i_plus1}.0.weight"].numpy()
    new[f"encoder.layers.{i}.norm2.bias"] = old[f"transformer.attn_layers.layers.{double_i_plus1}.0.bias"].numpy()
    new[f"encoder.layers.{i}.linear1.weight"] = old[f"transformer.attn_layers.layers.{double_i_plus1}.1.net.0.0.weight"].t().numpy()
    new[f"encoder.layers.{i}.linear1.bias"] = old[f"transformer.attn_layers.layers.{double_i_plus1}.1.net.0.0.bias"].numpy()
    new[f"encoder.layers.{i}.linear2.weight"] = old[f"transformer.attn_layers.layers.{double_i_plus1}.1.net.2.weight"].t().numpy() 
    new[f"encoder.layers.{i}.linear2.bias"] = old[f"transformer.attn_layers.layers.{double_i_plus1}.1.net.2.bias"].t().numpy() 

new["final_layer_norm.weight"] = old["transformer.norm.weight"].numpy()
new["final_layer_norm.bias"] = old["transformer.norm.bias"].numpy()

paddle.save(new, "./laion400M_pretrain/ldmbert/model_state.pdparams")