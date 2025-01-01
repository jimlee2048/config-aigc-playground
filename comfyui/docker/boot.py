import os
import subprocess
import shutil
import tomlkit
import git
import giturlparse
from comfy_cli.workspace_manager import WorkspaceManager
from comfy_cli.config_manager import ConfigManager
import comfy_cli.constants as cli_constants
import comfy_cli.cmdline as cli_cmd
import comfy_cli.command.custom_nodes.command as cli_cmd_node 

DEBUG = os.environ.get('DEBUG', False)
HF_API_TOKEN = os.environ.get('HF_API_TOKEN', None)
CIVITAI_API_TOKEN = os.environ.get('CIVITAI_API_TOKEN', None)
COMFYUI_EXTRA_ARGS = os.environ.get('COMFYUI_EXTRA_ARGS', None)
BOOT_CN_NETWORK = os.environ.get('BOOT_CN_NETWORK', False)
BOOT_CONFIG_DIR = os.environ.get('BOOT_CONFIG_DIR', None)
BOOT_INIT_NODE = os.environ.get('BOOT_INIT_NODE', False)
BOOT_INIT_MODEL = os.environ.get('BOOT_INIT_MODEL', False)
COMFYUI_PATH = os.environ.get('COMFYUI_PATH', "/workspace/ComfyUI")

def load_boot_config(path: str) -> dict:
    if not os.path.isdir(path): 
        print(f"Invalid config path: {path}")
        return {}
    config_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".toml"):
                config_files.append(os.path.join(root, file))
    print(f"Found {len(config_files)} config files in {path}:\n{config_files}")
    boot_config = {}
    for file in config_files:
        with open(file, 'rb') as f:
            boot_config.update(tomlkit.load(f))
    return boot_config

def init_node(boot_config: dict):
    node_config = boot_config.get('custom_nodes', {})
    all_count = len(node_config)
    print(f"Found {all_count} custom nodes in boot config, installing...")
    current_count = 0
    for node in node_config:
        current_count += 1
        node_url = node.get('url', '')
        if not node_url or giturlparse.parse(node_url).valid == False:
            print(f"[{current_count}/{all_count}] URL {node_url} invaild, skip...")
            continue
        node_name = giturlparse.parse(node_url).name
        if check_node_exists(node_name, COMFYUI_PATH):
            print(f"[{current_count}/{all_count}] {node_name} already exists, skip...")
            continue
        print(f"[{current_count}/{all_count}] Installing custom node: {node_name}")
        try:
            cli_cmd_node.install(nodes=[node_url],mode="remote")
        except Exception as e:
            print(f"Error installing {node_name}:\n{str(e)}")
            continue
        node_script = node.get('script', '')
        if node_script:
            exec_script(node_script)

def check_node_exists(node_name: str, comfyui_path: str) -> bool:
    if not os.path.isdir(comfyui_path):
        raise Exception(f"Invalid ComfyUI path: {comfyui_path}")
    node_path = os.path.join(comfyui_path, "custom_nodes", node_name)
    try:
        _ = git.Repo(node_path).git_dir
        return True
    except git.exc.InvalidGitRepositoryError:
        # remove invalid repo
        print(f"Invalid custom node repo: {node_name}, removing...")
        shutil.rmtree(node_path)
        return False
    except git.exc.NoSuchPathError:
        return False
    except Exception as e:
        print(f"Error checking custom node {node_name}:\n{str(e)}")
        return False

def exec_script(script_path: str) -> bool:
    if not os.path.isfile(script_path):
        print(f"Invalid script path: {script_path}")
        return False
    try:
        print(f"Executing script: {script_path}")
        result = subprocess.run(script_path, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running script '{script_path}':\n{result.stderr}")
            return False
        print(f"Successfully executed: {script_path}")
        return True
    except Exception as e:
        print(f"Exception while running script '{script_path}': {str(e)}")
        return False


if __name__ == '__main__':

    ## check if comfyui path exists
    if not os.path.isdir(COMFYUI_PATH):
        print(f"Invalid ComfyUI path: {COMFYUI_PATH}")
        exit(1)

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
            init_node(boot_config)

    launch_args_list = ["--listen", "0.0.0.0,::", "--port", "8188"] + (COMFYUI_EXTRA_ARGS.split() if COMFYUI_EXTRA_ARGS else [])
    launch_args_str = " ".join(launch_args_list).strip()
    cli_config_manager.set(cli_constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, launch_args_str)
    
    cli_cmd.env()
    cli_cmd.launch(background=False, extra=launch_args_list)