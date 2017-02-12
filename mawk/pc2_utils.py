from os.path import join, split, isfile, isdir
import subprocess as sp
from subprocess import call
import time


def build_shell_script_lines(path, job_name, res_plan, python_line):
    lines = ['#! /usr/bin/env zsh', '#! /bin/zsh', '',
             '#CCS -t ' + res_plan['duration'],
             '#CCS -o ' + join(path, '%reqid.log'),
             '#CCS -N ' + job_name,
             '#CCS --res=rset=' + res_plan['rset'] +
             ':ncpus=' + res_plan['ncpus'] +
             ':mem=' + res_plan['mem'] +
             ':vmem=' + res_plan['vmem'],
             '#CCS -j', '', python_line]
    return [line + '\n' for line in lines]


def calculate_resources(n_hl, n_units, seq_len):
        plan = {'duration': str(2*(1+min(2, (n_units*n_hl)//100))) + 'h',
                'rset': '1',
                'ncpus': str(2*(1 + n_hl*n_units//310)),
                'mem': str(min(max(1, n_hl*int(seq_len*4/7800))*12 +
                               (n_units//100)*8, 26))+'g',
                'vmem': str(min(max(1, n_hl*int(seq_len*5/7800))*12 +
                                (n_units//100)*8, 32))+'g'
                }
        return plan


def create_n_run_script(name, content, dry=False):
        with open(name, 'w+') as f:
            f.writelines(content)
        call(["chmod", "+x", name])  # Make script executable
        if not dry:
            call(['ccsalloc', name])  # allocate and run
            time.sleep(1)

if __name__ == "__main__":
    pass
