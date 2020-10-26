import socket
import threading
import time

numDrones = int(input("Enter Number of Drones in Fleet: "))

telloAddresses = []
localAddresses = []
sockets = []

for i in range(numDrones):
  # Define IP addresses of Tello EDUs starting at 101.
  telloAddresses.append(('192.168.0.10' + str(i+1), 8889))
  # Define IP and ports of computer.
  localAddresses.append(('', int(str(901) + str(i))))
  # Create UDP connection to send commands via.
  sockets.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))
  # Bind sockets to the corresponding local address and port.
  sockets[i].bind(localAddresses[i])

# Send command to Tello drones and allow for a delay in seconds.
def send(cmd, delay):
  # Try to send the command to Tello drones, else print an exception.
  try:
    for i in range(numDrones):
      sockets[i].sendto(cmd.encode(), telloAddresses[i])
    print("Sending command: " + cmd)
  except Exception as e:
    print("Error sending: " + str(e))

  # Delay next action for a specified period of time.
  time.sleep(delay)

# Receive responses from Tello drones.
def receive():
  # Continuously loop and listen for incoming responses.
  while True:
    # Try to receive the response, else print the exception.
    try:
      responses = []
      for i in range(numDrones):
        responses.append('')
        responses[i], ipAddress = sockets[i].recvfrom(128)
        print("Received response, Drone " + str(i+1) + ": " + responses[i].decode(encoding='utf-8'))
    except Exception as e:
      # If there's an error close the sockets and break out of the loop.
      for i in range(numDrones):
        sockets[i].close()
      print("Error receiving: " + str(e))
      break

# Create and start a listening thread that runs in the background.
# This will continuously monitor for incoming responses through receive().
receiveThread = threading.Thread(target=receive)
receiveThread.daemon = True
receiveThread.start()

# Put Tello EDUs into command mode.
# Right now, this is just used to verify that the Tello EDU drones were successfully connected.
send("command", 3)

# Notify user that drones are ready for commands.
print("Ready.")

# Close sockets once all commands are complete.
# FIXME: Throwing an error at end because sockets are closed, but thread is still running.
for i in range(numDrones):
  sockets[i].close()