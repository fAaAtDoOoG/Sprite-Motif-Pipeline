from sprite_motif_pipeline.workflow import build_api_prompt, required_node_types


def test_workflow_contains_required_nodes_and_lora():
    prompt = build_api_prompt(
        positive_prompt="Pixel Art, character",
        negative_prompt="blur",
        width=1024,
        height=1024,
        seed=1,
        filename_prefix="test",
    )
    classes = {node["class_type"] for node in prompt.values()}
    assert required_node_types().issubset(classes)
    assert prompt["4"]["inputs"]["lora_name"].endswith("Pixel-Art-LoRA.safetensors")
    assert prompt["8"]["inputs"]["width"] == 1024
    assert prompt["8"]["inputs"]["height"] == 1024
