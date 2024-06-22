import os
import paramiko

def create_ssh_tunnel(jump_host, target_host, private_key_path, jump_port=22, target_port=22):
    # Connect to the jump host
    jump_transport = paramiko.Transport((jump_host, jump_port))
    jump_transport.connect(username="gclyne", pkey=private_key_path)

    # Create an SSH tunnel to the target host via the jump host
    target_transport = jump_transport.open_channel('direct-tcpip', (target_host, target_port))
    return target_transport


def scp_copy(src_path, dest_path, hostname,port=22):
    # Connect to the remote server using SCP
    private_key = paramiko.RSAKey(filename='/Users/gclyne/.ssh/id_rsa')

    #     # Connect to the source remote server using SCP
    src_transport = paramiko.Transport(('spiritx1.ipsl.fr', port))
    src_transport.connect(username="gclyne", pkey=private_key)
    src_scp = paramiko.SFTPClient.from_transport(src_transport)
    print(src_scp)
    # Connect to the destination remote server using SCP
    
    dest_transport = create_ssh_tunnel('ssh-pro.inria.fr', 'jean-zay.idris.fr', private_key, 22, 22).get_transport()
    print(dest_transport)
    # print(dest_transport.get_transport())
    dest_scp = paramiko.SFTPClient.from_transport(dest_transport)
    print(dest_transport)
    dest_scp = paramiko.SFTPClient.from_transport(dest_transport)
    print(dest_scp)
    for path in src_scp.listdir(src_path):
        print(path)
        for group in src_scp.listdir(src_path + path):
            print(group)
            if(group in ['Amon','Emon','Lmon']):
                for var_path in src_scp.listdir(src_path + path + '/' + group):
                    print(var_path)
                    for file_path in src_scp.listdir(src_path + path + '/' + group + '/' + var_path + '/gr/latest/'):
                        print(file_path)
                        src_file_path = src_path + path + '/' + group + '/' + var_path + '/gr/latest/' + file_path
                        src_scp.get(src_path + path + '/' + group + '/' + var_path + '/gr/latest/' + file_path, dest_path + path + '/' + group + '/' + var_path + '/' + file_path)
                        dest_file_path = os.path.join(dest_path, path,group,var_path)
                        src_scp.get(src_file_path, dest_file_path)
        break

    # Close the SCP connection
    src_scp.close()
    src_transport.close()
    dest_scp.close()
    dest_transport.close()

if __name__ == "__main__":
    # Set your source and destination directories
    experiment = "dcppA-hindcast"
    source_directory =f'/bdd/CMIP6/DCPP/IPSL/IPSL-CM6A-LR/{experiment}/'
    destination_directory = f'/gpfsstore/rech/mlr/udy16au/{experiment}'

    # Set your remote server credentials
    remote_port = 22  # Default SSH port

    # Perform SCP copy
    scp_copy(source_directory, destination_directory,remote_port)

