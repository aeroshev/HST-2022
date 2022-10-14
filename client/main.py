import socket

HOST = '127.0.0.1'
PORT = 8889


if __name__ == '__main__':
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((HOST, PORT))
        sock.sendall(b"Hello, world!\n")
        response = sock.recv(1024)

    print(f"C server response - {response}")
