
import subprocess
import os

def set_proxy(proxy_url, http_port, socks_port=None):
    print("\n🟢 Setting up proxy...")

    http_proxy = f"http://{proxy_url}:{http_port}"
    os.environ['HTTP_PROXY'] = http_proxy
    os.environ['HTTPS_PROXY'] = http_proxy
    os.environ['http_proxy'] = http_proxy
    os.environ['https_proxy'] = http_proxy
    print(f"🟢 HTTP_PROXY: {http_proxy}"
          f"\n🟢 HTTPS_PROXY: {http_proxy}")
    
    if socks_port:
        socks_proxy = f"socks5://{proxy_url}:{socks_port}"
        os.environ['SOCKS_PROXY'] = socks_proxy
        os.environ['socks_proxy'] = socks_proxy
        print(f"🟢 SOCKS_PROXY: {socks_proxy}")
    
    no_proxy = "localhost,*.local,*.internal,[::1],fd00::/7,10.0.0.0/8,127.0.0.0/8,169.254.0.0/16,172.16.0.0/12,192.168.0.0/16,10.*,127.*,169.254.*,172.16.*,172.17.*,172.18.*,172.19.*,172.20.*,172.21.*,172.22.*,172.23.*,172.24.*,172.25.*,172.26.*,172.27.*,172.28.*,172.29.*,172.30.*,172.31.*,172.32.*,192.168.*,*.cn,ghproxy.com,*.ghproxy.com,ghproxy.org,*.ghproxy.org,gh-proxy.com,*.gh-proxy.com,ghproxy.net,*.ghproxy.net"
    os.environ['NO_PROXY'] = no_proxy
    os.environ['no_proxy'] = no_proxy
    print(f"🟢 NO_PROXY: {no_proxy}")


    print("\n🟢 Proxy has been set successfully.")
    return True



def set_proxy_autodl():
    print("\n🟢 Setting up proxy...")
    result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"',
                            shell=True, capture_output=True, text=True)
    output = result.stdout
    for line in output.splitlines():
        if '=' in line:
            var, value = line.split('=', 1)
            os.environ[var] = value
            print(f"🟢 {var}: {value}")
    return True
