/*
 * stack.c
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
 *     but WITHOUT ANY WARRANTY without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *     GNU General Public License for more details.
 * 
 *     You should have received a copy of the GNU General Public License
 *     along with TDDD56. If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef DEBUG
#define NDEBUG
#endif

#include <assert.h>
#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "stack.h"
#include "non_blocking.h"

#if NON_BLOCKING == 0
#warning Stacks are synchronized through locks
#else
#if NON_BLOCKING == 1
#warning Stacks are synchronized through hardware CAS
#else
#warning Stacks are synchronized through lock-based CAS
#endif
#endif

int
stack_check(stack_t *stack)
{
// Do not perform any sanity check if performance is bein measured
#if MEASURE == 0
	// Use assert() to check if your stack is in a state that makes sens
	// This test should always pass 
	assert(1 == 1);

	// This test fails if the task is not allocated or if the allocation failed
	assert(stack != NULL);
#endif
	// The stack is always fine
	return 1;
}

void simple_push(simple_stack_t* stack, node_t* newNode) {
  // node_t* newNode = (node_t*)malloc(sizeof(node_t));
  // newNode->data = value;
  // newNode->next = NULL;

  newNode->next = stack->head->next;
  stack->head->next = newNode;
}

node_t* simple_pop(simple_stack_t* stack) {
  node_t* toBeReturned = stack->head->next;
  if(toBeReturned != NULL)
    stack->head->next = stack->head->next->next;
  return toBeReturned;
}

node_t* getNode(stack_t* stack, int value, int id) {
  node_t* newNode = simple_pop(&stack->pool[id]);
  if(newNode == NULL) {
    newNode = (node_t*)malloc(sizeof(node_t));
  }
  newNode->data = value;
  newNode->next = NULL;
  return newNode;
}

void /* Return the type you prefer */
stack_push(stack_t* stack, pthread_mutex_t* mutex, int value, int id)
{
node_t* newNode = getNode(stack, value, id);

#if NON_BLOCKING == 0
  pthread_mutex_lock(mutex); // Critical section here
  newNode->next = stack->head->next;
  stack->head->next = newNode;
  pthread_mutex_unlock(mutex);

#elif NON_BLOCKING == 1
  // inline size_t cas(size_t* reg, size_t oldval, size_t newval)
  // Implement a harware CAS-based stack
  node_t* old;
  do {
    old = stack->head->next;
    newNode->next = old;
  } while(cas((size_t*)&stack->head->next, (size_t)old, (size_t)newNode) != (size_t)old);

  
#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  // Debug practice: you can check if this operation results in a stack in a consistent check
  // It doesn't harm performance as sanity check are disabled at measurement time
  // This is to be updated as your implementation progresses
  stack_check((stack_t*)1);
}

int /* Return the type you prefer */
stack_pop(stack_t* stack, pthread_mutex_t* mutex, int id)
{
#if NON_BLOCKING == 0
  // Implement a lock_based stack
  int dataToReturn = stack->head->next->data;

  pthread_mutex_lock(mutex); // Critical section here
  node_t* toBeRemoved = stack->head->next;
  stack->head->next = stack->head->next->next;
  pthread_mutex_unlock(mutex);

  simple_push(&stack->pool[id], toBeRemoved);
  return dataToReturn;

#elif NON_BLOCKING == 1
  // Implement a harware CAS-based stack

  node_t* toBeRemoved;
  node_t* newHead;
  do {
    toBeRemoved = stack->head->next;
    newHead = stack->head->next->next;
  } while(cas((size_t*)&stack->head->next, (size_t)toBeRemoved, (size_t)newHead) != (size_t)toBeRemoved);

  simple_push(&stack->pool[id], toBeRemoved);
  return toBeRemoved->data;

#else
  /*** Optional ***/
  // Implement a software CAS-based stack
#endif

  return 0;
}

#if NON_BLOCKING == 1 || NON_BLOCKING == 2
void* ABA_slow_pop(stack_t* stack, pthread_mutex_t* mutex, int id) {
  printf("\nsaving pointers\n");
  node_t* toBeRemoved = stack->head->next;
  node_t* newHead = stack->head->next->next;

  pthread_mutex_lock(mutex);
  printf("thread 0 swapping\n");
  cas((size_t*)&stack->head->next, (size_t)toBeRemoved, (size_t)newHead);
  simple_push(&stack->pool[id], toBeRemoved);
  pthread_mutex_unlock(mutex);
}

void* pool_wait_pop(stack_t* stack, pthread_mutex_t* mutex, int id) {
  node_t* toBeRemoved;
  node_t* newHead;
  do {
    toBeRemoved = stack->head->next;
    newHead = stack->head->next->next;
  } while(cas((size_t*)&stack->head->next, (size_t)toBeRemoved, (size_t)newHead) != (size_t)toBeRemoved);

  pthread_mutex_lock(mutex);
  printf("Thread one pushing to pool\n");
  simple_push(&stack->pool[id], toBeRemoved);
  pthread_mutex_unlock(mutex);
  return toBeRemoved->data;
}
#endif

