#ifndef TCP_SERVER_H
#define TCP_SERVER_H

#include <stdio.h>
#include <netdb.h>
#include <string.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#define MAX 1024
#define SA struct sockaddr


void worker(int connfd, int (*func)(int**, unsigned int));
void serve(unsigned short port, int (*func)(int**, unsigned int));

#endif
