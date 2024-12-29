import os
from comfy_cli import constants
from comfy_cli.config_manager import ConfigManager
from comfy_cli.workspace_manager import WorkspaceManager
import comfy_cli.cmdline as comfy_cli_cmd


HF_API_TOKEN = os.environ.get('HF_API_TOKEN', None)
CIVITAI_API_TOKEN = os.environ.get('CIVITAI_API_TOKEN', None)
COMFYUI_EXTRA_ARGS = os.environ.get('COMFYUI_EXTRA_ARGS', None)


if __name__ == '__main__':

    config_manager = ConfigManager()
    if HF_API_TOKEN:
        config_manager.set(constants.HF_API_TOKEN_KEY, HF_API_TOKEN)
    if CIVITAI_API_TOKEN:
        config_manager.set(constants.CIVITAI_API_TOKEN_KEY, CIVITAI_API_TOKEN)

    launch_args_list = ["--listen", "0.0.0.0,::", "--port", "8188"] + (COMFYUI_EXTRA_ARGS.split() if COMFYUI_EXTRA_ARGS else [])
    launch_args_str = " ".join(launch_args_list).strip()
    config_manager.set(constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, launch_args_str)
    
    workspace_manager = WorkspaceManager()
    workspace_manager.setup_workspace_manager(
        specified_workspace=None, # Use default workspace
        use_here=False,
        use_recent=False, 
        skip_prompting=True
    )
    comfy_cli_cmd.env()
    comfy_cli_cmd.launch(background=False, extra=launch_args_list)