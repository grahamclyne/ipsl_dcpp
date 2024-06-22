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
                    if any(substring in variant for substring in [str(x) for x in list(range(1960,1992))]):
                        continue
                    for group in source_conn.run(f'ls {source_path + variant}',hide=True).stdout.rsplit('\n'):
                        if(group == ''):
                            continue
                        if(group in ['Omon']):
                            
                            for var_path in source_conn.run(f'ls {source_path + variant}/{group}',hide=True).stdout.rsplit('\n'):
                                if(var_path == ''):
                                    continue
                                for sub_var_path in source_conn.run(f'ls {source_path + variant}/{group}/{var_path}',hide=True).stdout.rsplit('\n'):
                                    if(sub_var_path == ''):
                                        continue
                                    for file_path in source_conn.run(f'ls {source_path + variant}/{group}/{var_path}/{sub_var_path}/latest/',hide=True).stdout.rsplit('\n'):
                                        if(file_path == ''):
                                            continue    
    
                                        # list_of_variables_to_exclude = ['tsl','solth','expfe','ppdiat','ppmisc','tdps','fracLut','nwdFracLut','mrsfl','mrsll','mrsol','solay','olevel_bounds','time_bounds','area','bounds_nav_lon','bounds_nav_lat','olevel','lev','flandice','t20d','thetaot','thetaot2000','thetaot300','thetaot700','nav_lat','nav_lon']
                                        # if (len([ele for ele in list_of_variables_to_exclude if(ele in file_path)]) > 0):
                                        #     print('skipping')
                                        #     continue
                                        if(file_path.startswith('tos')):
                                            print(file_path)
                                            dest_file_path = os.path.join(destination_path, variant,group)
                                            print('dest_file_path',dest_file_path)
                                            try:
                                                ls_result = dest_conn.run(f'ls {dest_file_path}',hide=True).stdout
                                                if(file_path in ls_result):
                                                    print('zipped file already exists')
                                                    continue
                                            except:
                                                dest_conn.run(f'mkdir -p {dest_file_path}')
                                            src_file_path = f'{source_path + variant}/{group}/{var_path}/{sub_var_path}/latest/{file_path}'
                                        else:
                                            continue
                                #process each group of files
                                #-j option leaves out the path in the zip file
                                        print(f'transferring {src_file_path} to {dest_file_path}')
                                        dest_conn.run(f'scp -i ~/.ssh/priv_rsa {source_host}:{src_file_path} {dest_file_path}')

if __name__ == "__main__":
    # Set your jump host, source host, destination host, paths, and private key path
    jump_host = "gclyne@ssh-pro.inria.fr"
    source_host = "gclyne@spiritx1.ipsl.fr"
    experiment = "dcppA-hindcast"
    source_path =f'/bdd/CMIP6/DCPP/IPSL/IPSL-CM6A-LR/{experiment}/'
    destination_host = "udy16au@jean-zay.idris.fr"
    destination_path = "/gpfsstore/rech/mlr/udy16au"
    private_key_path = "/Users/gclyne/.ssh/id_rsa"

    # Perform SCP copy with jump connection using Fabric
    scp_copy_with_jump(jump_host, source_host, source_path, destination_host, destination_path, private_key_path)
