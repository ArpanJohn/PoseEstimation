import subprocess

scripts = ['rec_program\\rec_to_msg_pcmp_que.py', 'rec_program\\rec_to_msg_pcmp_rsq.py']

for script in scripts:
    print('running', script)
    subprocess.run(['python', script])
