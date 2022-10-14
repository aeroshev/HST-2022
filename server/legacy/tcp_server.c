#include "tcp_server.h"


void worker(int connfd, int (*func)(int**, unsigned int)) {
    char buffer[MAX];
    memset(&buffer, 0, MAX);

    read(connfd, &buffer, sizeof(buffer));
    fprintf(stdout, "Received message from client\n");

    unsigned int n = 0;
    while ((buffer[n++] = getchar()) != '\n')
        ;
    
    
    write(connfd, buffer, sizeof(buffer));
}


void serve(unsigned short port, int (*func)(int**, unsigned int)) {
    int sockfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (sockfd == -1) {
        fprintf(stderr, "Socket creation failed...\n");
        exit(1);
    }
    else {
        fprintf(stdout,"Socket successfully created..\n");
    }

    struct sockaddr_in servaddr;
    memset(&servaddr, 0, sizeof(servaddr));

    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(port);

    if ((bind(sockfd, (SA *)&servaddr, sizeof(servaddr))) != 0) {
        fprintf(stderr, "Socket bind failed...\n");
        exit(1);
    } else {
        fprintf(stdout, "Socket successfully binded..\n");
    }

    if ((listen(sockfd, 5)) != 0 ) {
        fprintf(stderr, "Listen failed...\n");
        exit(1);
    } else {
        fprintf(stdout, "Server listening..\n");
    }

    int connfd = 0;
    struct sockaddr_in cli;
    unsigned int len = sizeof(cli);
    while(1) {
        fprintf(stdout, "Ready for accept request\n");

        connfd = accept(sockfd, (SA*)&cli, &len);
        if (connfd < 0)
            fprintf(stderr, "Server accept failed...\n");
        else
            fprintf(stdout, "Server accept the client...\n");
        
        worker(connfd, func);

        close(connfd);
        fprintf(stdout, "Closed connection\n");
    }

    return;
}
