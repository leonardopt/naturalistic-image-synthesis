"""
GPU thermal monitor for long generation runs.

Polls nvidia-smi every second. If any GPU exceeds 90°C the script enters a
hold loop (checking every 5 s) and resumes only once all GPUs have cooled
below 50°C. Run in a separate terminal alongside pipe_2 or pipe_5 to prevent
thermal throttling or hardware damage during multi-GPU generation jobs.
"""
import subprocess
import time


def get_gpu_temperatures():
    temperatures = []
    try:
        # Run nvidia-smi command to get temperatures of all GPUs
        result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                                stdout=subprocess.PIPE)
        # Decode the output from bytes to string and split by newlines to get each GPU's temperature
        output = result.stdout.decode('utf-8').strip().split('\n')
        # Convert each temperature string to int and append to temperatures list
        temperatures = [int(temp) for temp in output]
    except Exception as e:
        print(f"Failed to get GPU temperatures: {e}")
    return temperatures


def main():
    high_temperature_threshold = 90  # Temperature threshold to pause
    resume_temperature_threshold = 50  # Temperature threshold to resume
    while True:
        temps = get_gpu_temperatures()
        if temps:
            # Check if any GPU's temperature exceeds the high threshold
            if any(temp >= high_temperature_threshold for temp in temps):
                print("Temperature exceeds threshold, pausing...")
                while True:
                    time.sleep(5)  # Check temperatures every 5 seconds during pause
                    temps = get_gpu_temperatures()
                    print(f"Current temperatures during pause: {temps}")
                    # Resume if all GPUs have cooled down to the resume threshold
                    if all(temp <= resume_temperature_threshold for temp in temps):
                        print("Temperatures have dropped, resuming operations...")
                        break
            else:
                print(f"Current temperatures: {temps}")
        else:
            print("Could not get GPU temperatures.")

        time.sleep(1)  # Check temperatures every 1 second


if __name__ == "__main__":
    main()
