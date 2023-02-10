from pssh.clients import ParallelSSHClient
import subprocess
import socket
import os

def get_ips_of_slurm_job(job_id):
    c = "sinfo -N -n `squeue -j "+str(job_id)+" | tail -1 | awk '{print $8}'` | tail -n +2 | awk '{print $1}'"
    hosts = subprocess.check_output(c, shell=True).decode("utf8")[:-1].split("\n")
    ips = [socket.gethostbyname(host) for host in hosts]

    return ips

def run(ips_to_run, command):
    print(ips_to_run)

    client = ParallelSSHClient(ips_to_run)
    output = list(client.run_command(command, stop_on_errors=True))

    print(["\n".join(o.stdout) for o in output])