import os
import subprocess
import shutil
from pathlib import Path
import tomlkit
import git
import giturlparse
from comfy_cli.workspace_manager import WorkspaceManager
from comfy_cli.config_manager import ConfigManager
import comfy_cli.constants as cli_constants
import comfy_cli.cmdline as cli_cmd
import comfy_cli.command.custom_nodes.command as cli_cmd_node 
import comfy_cli.command.models.models as cli_cmd_model

DEBUG = os.environ.get('DEBUG', False)
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', None)
CIVITAI_API_TOKEN = os.environ.get('CIVITAI_API_TOKEN', None)
COMFYUI_EXTRA_ARGS = os.environ.get('COMFYUI_EXTRA_ARGS', None)
BOOT_CN_NETWORK = os.environ.get('BOOT_CN_NETWORK', False)
BOOT_CONFIG_DIR = os.environ.get('BOOT_CONFIG_DIR', None)
BOOT_INIT_NODE = os.environ.get('BOOT_INIT_NODE', False)
BOOT_INIT_MODEL = os.environ.get('BOOT_INIT_MODEL', False)
COMFYUI_PATH = Path(os.environ.get('COMFYUI_PATH', "/workspace/ComfyUI"))

def load_boot_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_dir():
        print(f"Invalid config path: {path}")
        if config_path.is_file():
            config_path.unlink()
        config_path.mkdir(parents=True, exist_ok=True)
        return {}
    
    config_files = list(config_path.rglob("*.toml"))
    print(f"Found {len(config_files)} config files in {path}:\n{config_files}")
    
    boot_config = {}
    for file in config_files:
        boot_config.update(tomlkit.loads(file.read_text()))
    return boot_config

def is_valid_git_repo(path: str) -> bool:        
    try:
        _ = git.Repo(path).git_dir
        return True
    except Exception as e:
        return False

def init_nodes(boot_config: dict):
    node_config = boot_config.get('custom_nodes', {})
    all_count = len(node_config)
    print(f"Found {all_count} custom nodes in boot config")
    
    for current_count, node in enumerate(node_config, 1):
        try:
            node_url = node['url']
            node_name = giturlparse.parse(node_url).name
            if not giturlparse.parse(node_url).valid:
                raise Exception(f"Invalid git URL")
        except Exception as e:
            print(f"[{current_count}/{all_count}] Invalid node config: {node}\n{str(e)}")
            continue
        
        node_path = COMFYUI_PATH / "custom_nodes" / node_name
        if node_path.exists and node_path.is_dir():
            if is_valid_git_repo(node_path):
                print(f"[{current_count}/{all_count}] {node_name} already exists, skip...")
                continue
            else:
                print(f"[{current_count}/{all_count}] {node_name} broken, removing...")
                shutil.rmtree(node_path)

        print(f"[{current_count}/{all_count}] Installing custom node: {node_name}")
        try:
            cli_cmd_node.install(nodes=[node_url], mode="remote")
        except Exception as e:
            print(f"Error installing {node_name}:\n{str(e)}")
            continue

        node_script = node.get('script', '')
        if node_script:
            exec_script(node_script)

def exec_script(path: str) -> bool:
    script = Path(path)
    if not script.is_file():
        print(f"Invalid script path: {path}")
        return False
        
    try:
        print(f"Executing script: {path}")
        result = subprocess.run(str(script), shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running script '{path}':\n{result.stderr}")
            return False
        print(f"Successfully executed: {path}")
        return True
    except Exception as e:
        print(f"Exception while running script '{path}': {str(e)}")
        return False

def init_models(boot_config: dict):
    model_config = boot_config.get('models', {})
    all_count = len(model_config)
    print(f"Found {all_count} models in boot config")
    
    for current_count, model in enumerate(model_config, 1):
        try:
            model_url = model['url']
            model_dir = model['dir']
            model_filename = model['filename']
        except Exception as e:
            print(f"[{current_count}/{all_count}] Invalid model config: {model}\n{str(e)}")
            continue
        model_path = COMFYUI_PATH / model_dir / model_filename
        if model_path.exists and model_path.is_file():
            print(f"[{current_count}/{all_count}] {model_filename} already exists, skip...")
            continue
        else:
            print(f"[{current_count}/{all_count}] {model_filename} broken, removing...")
            shutil.rmtree(model_path)
        print(f"[{current_count}/{all_count}] Downloading model: {model_filename}")
        try:
            cli_cmd_model.download(url=model_url, relative_path=model_dir, filename=model_filename)
        except Exception as e:
            print(f"Error downloading {model_filename}:\n{str(e)}")
            continue

if __name__ == '__main__':
    # check if comfyui path exists
    if not COMFYUI_PATH.is_dir():
        raise Exception(f"ERROR: Invalid ComfyUI path \"{COMFYUI_PATH}\"")

    # chinese mainland network settings
    if BOOT_CN_NETWORK:
        print("Optimizing for Chinese Mainland network...")
        # pip source to ustc mirror
        os.environ['PIP_INDEX_URL'] = 'https://mirrors.ustc.edu.cn/pypi/web/simple'
        # huggingface endpoint to hf-mirror.com
        os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
        # TODO: civitai, git, apt, etc.

    ## necessary setup for comfy_cli
    cli_config_manager = ConfigManager()
    cli_workspace_manager = WorkspaceManager()
    cli_workspace_manager.setup_workspace_manager(specified_workspace=None, use_here=False, use_recent=False, skip_prompting=True)

    if HF_API_TOKEN:
        cli_config_manager.set(cli_constants.HF_API_TOKEN_KEY, HF_API_TOKEN)
    if CIVITAI_API_TOKEN:
        cli_config_manager.set(cli_constants.CIVITAI_API_TOKEN_KEY, CIVITAI_API_TOKEN)

    boot_config = load_boot_config(BOOT_CONFIG_DIR)
    if not boot_config:
        print("No boot config found, continue with default settings...")
    else:
        if BOOT_INIT_NODE:
            init_nodes(boot_config)
        if BOOT_INIT_MODEL:
            init_models(boot_config)

    launch_args_list = ["--listen", "0.0.0.0,::", "--port", "8188"] + (COMFYUI_EXTRA_ARGS.split() if COMFYUI_EXTRA_ARGS else [])
    launch_args_str = " ".join(launch_args_list).strip()
    cli_config_manager.set(cli_constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, launch_args_str)
    
    cli_cmd.env()
    cli_cmd.launch(background=False, extra=launch_args_list)