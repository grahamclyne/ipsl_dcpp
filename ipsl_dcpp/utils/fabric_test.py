from fabric import Connection
import os

def scp_copy_with_jump(jump_host, source_host, source_path, destination_host, destination_path, private_key_path):
    # Connect to the jump host
    with Connection(jump_host, connect_kwargs={"key_filename": private_key_path}) as jump_conn:
        # Connect to the source host via the jump host
        print(jump_conn)
        with Connection(source_host, connect_kwargs={"key_filename": private_key_path}) as source_conn:
            # Connect to the destination host via the jump host
            print(source_conn)
            with Connection(destination_host, connect_kwargs={"key_filename": private_key_path}, gateway=jump_conn) as dest_conn:
                # Create the remote directory on the destination host if it doesn't exist
                print(dest_conn)                
                for variant in source_conn.run(f'ls {source_path}',hide=True).stdout.rsplit('\n'):
                    print(variant)
                    if(variant == ''):
                        continue
                    for group in source_conn.run(f'ls {source_path + variant}',hide=True).stdout.rsplit('\n'):
                        if(group == ''):
                            continue
                        if(group in ['Amon','Emon','Lmon']):
                            archive_name = f'archive_{variant}_{group}'
                            dest_file_path = os.path.join(destination_path, variant,group)
                            print('dest_file_path',dest_file_path)
                            try:
                                ls_result = dest_conn.run(f'ls {dest_file_path}',hide=True).stdout
                                if(archive_name in ls_result):
                                    print('zipped file already exists')
                                    continue
                            except:
                                dest_conn.run(f'mkdir -p {dest_file_path}')
                            for var_path in source_conn.run(f'ls {source_path + variant}/{group}',hide=True).stdout.rsplit('\n'):
                                if(var_path == ''):
                                    continue
                                for sub_var_path in source_conn.run(f'ls {source_path + variant}/{group}/{var_path}',hide=True).stdout.rsplit('\n'):
                                    if(sub_var_path == ''):
                                        continue
                                    for file_path in source_conn.run(f'ls {source_path + variant}/{group}/{var_path}/{sub_var_path}/latest/',hide=True).stdout.rsplit('\n'):
                                        print(file_path)
                                        # list_of_variables_to_exclude = ['tsl','solth','expfe','ppdiat','ppmisc','tdps','fracLut','nwdFracLut','mrsfl','mrsll','mrsol','solay','olevel_bounds','time_bounds','area','bounds_nav_lon','bounds_nav_lat','olevel','lev','flandice','t20d','thetaot','thetaot2000','thetaot300','thetaot700','nav_lat','nav_lon']
                                        # if (len([ele for ele in list_of_variables_to_exclude if(ele in file_path)]) > 0):
                                        #     print('skipping')
                                        #     continue
                                        src_file_path = f'{source_path + variant}/{group}/{var_path}/{sub_var_path}/latest/{file_path}'
                                        if(file_path == ''):
                                            continue                         
                                        source_conn.run(f'echo {src_file_path} >> /home/gclyne/source_paths.txt')
                            #process each group of files
                            #-j option leaves out the path in the zip file
                            source_conn.run(f'zip -j {archive_name}.zip -@ < /home/gclyne/source_paths.txt')
                            dest_conn.run(f'scp -i ~/.ssh/priv_rsa {source_host}:{archive_name}.zip {dest_file_path}')
                            source_conn.run(f'rm {archive_name}.zip')    
                            source_conn.run(f'rm /home/gclyne/source_paths.txt')
                            dest_conn.run(f'unzip {dest_file_path}/{archive_name}.zip -d {dest_file_path}')

if __name__ == "__main__":
    # Set your jump host, source host, destination host, paths, and private key path
    jump_host = "gclyne@ssh-pro.inria.fr"
    source_host = "gclyne@spiritx1.ipsl.fr"
    experiment = "dcppC-amv-neg"
    source_path =f'/bdd/CMIP6/DCPP/IPSL/IPSL-CM6A-LR/{experiment}/'
    destination_host = "udy16au@jean-zay.idris.fr"
    destination_path = f"/gpfsstore/rech/mlr/udy16au/{experiment}"
    private_key_path = "/Users/gclyne/.ssh/id_rsa"

    # Perform SCP copy with jump connection using Fabric
    scp_copy_with_jump(jump_host, source_host, source_path, destination_host, destination_path, private_key_path)
