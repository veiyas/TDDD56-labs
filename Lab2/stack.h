/*
 * stack.h
 *
 *  Created on: 18 Oct 2011
 *  Copyright 2011 Nicolas Melot
 *
 * This file is part of TDDD56.
 * 
 *     TDDD56 is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 * 
 *     TDDD56 is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include <stdlib.h>
#include <pthread.h>

#ifndef STACK_H
#define STACK_H

struct node {
  int data;
  struct node* next;
};
typedef struct node node_t;

struct stack
{
  node_t* head;
};
typedef struct stack stack_t;

void stack_push(stack_t* stack, pthread_mutex_t* mutex, int value);
int stack_pop(stack_t* stack, pthread_mutex_t* mutex);

/* Use this to check if your stack is in a consistent state from time to time */
int stack_check(stack_t *stack);
#endif /* STACK_H */
