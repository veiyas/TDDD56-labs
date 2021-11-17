/*
 * test.c
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

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <stddef.h>

#include "test.h"
#include "stack.h"
#include "non_blocking.h"

#define test_run(test)\
  printf("[%s:%s:%i] Running test '%s'... ", __FILE__, __FUNCTION__, __LINE__, #test);\
  test_setup();\
  if(test())\
  {\
    printf("passed\n");\
  }\
  else\
  {\
    printf("failed\n");\
  }\
  test_teardown();

/* Helper function for measurement */
double timediff(struct timespec *begin, struct timespec *end)
{
	double sec = 0.0, nsec = 0.0;
   if ((end->tv_nsec - begin->tv_nsec) < 0)
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec  - 1);
      nsec = (double)(end->tv_nsec - begin->tv_nsec + 1000000000);
   } else
   {
      sec  = (double)(end->tv_sec  - begin->tv_sec );
      nsec = (double)(end->tv_nsec - begin->tv_nsec);
   }
   return sec + nsec / 1E9;
}

typedef int data_t;
#define DATA_SIZE sizeof(data_t)
#define DATA_VALUE 5

#ifndef NDEBUG
int
assert_fun(int expr, const char *str, const char *file, const char* function, size_t line)
{
	if(!(expr))
	{
		fprintf(stderr, "[%s:%s:%zu][ERROR] Assertion failure: %s\n", file, function, line, str);
		abort();
		// If some hack disables abort above
		return 0;
	}
	else
		return 1;
}
#endif

stack_t *stack;
data_t data;
pthread_mutex_t mutex;

#if MEASURE != 0
struct stack_measure_arg
{
  int id;
};
typedef struct stack_measure_arg stack_measure_arg_t;

struct timespec t_start[NB_THREADS], t_stop[NB_THREADS], start, stop;

#if MEASURE == 1
void*
stack_measure_pop(void* arg)
  {
    stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
    int i;
    // for(int i = 0; i < MAX_PUSH_POP; ++i) {
    //   stack_push(stack, &mutex, i);
    // }

    clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
    for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
      {
        stack_pop(stack, &mutex);
      }
    clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

    return NULL;
  }
#elif MEASURE == 2
void*
stack_measure_push(void* arg)
{
  stack_measure_arg_t *args = (stack_measure_arg_t*) arg;
  int i;
  // for(int i = 0; i < MAX_PUSH_POP; ++i) {
  //   node_t* newNode = (node_t*)malloc(sizeof(node_t));
  //   newNode->data = -1;
  //   newNode->next = stack->pool.head->next;
  //   stack->pool.head->next = newNode;
  // }

  clock_gettime(CLOCK_MONOTONIC, &t_start[args->id]);
  for (i = 0; i < MAX_PUSH_POP / NB_THREADS; i++)
    {
        stack_push(stack, &mutex, i);
    }
  clock_gettime(CLOCK_MONOTONIC, &t_stop[args->id]);

  return NULL;
}
#endif
#endif


/* A bunch of optional (but useful if implemented) unit tests for your stack */
void
test_init()
{
  // Initialize your test batch
  pthread_mutex_init(&mutex, NULL );
}

void
test_setup()
{
  // Allocate and initialize your test stack before each test
  data = DATA_VALUE;

  // Allocate a new stack and reset its values
  stack = malloc(sizeof(stack_t));
  stack->head = (node_t*)malloc(sizeof(node_t));
  stack->head->data = -1;
  stack->head->next = NULL;

  stack->pool.head = (node_t*)malloc(sizeof(node_t));

  #if MEASURE == 0
  for(int i = 0; i < 10; ++i) {
      node_t* newNode = (node_t*)malloc(sizeof(node_t));
      newNode->data = -1;
      newNode->next = stack->pool.head->next;
      stack->pool.head->next = newNode;
    }
  #elif MEASURE == 1
    for(int i = 0; i < MAX_PUSH_POP; ++i) {
      stack_push(stack, &mutex, i);
    }
  #elif MEASURE == 2
    for(int i = 0; i < MAX_PUSH_POP; ++i) {
      node_t* newNode = (node_t*)malloc(sizeof(node_t));
      newNode->data = -1;
      newNode->next = stack->pool.head->next;
      stack->pool.head->next = newNode;
    }
  #endif

  // Reset explicitely all members to a well-known initial value
  // For instance (to be deleted as your stack design progresses):
}

// CAS: En tråd läser minnet som den vill modifiera innan den uppdaterar minnet. Om minnet har samma värde som rapporterats till tråden
// så genomför den uppdateringen vilket är en atomisk operation, oftast på hårdvarunivå. Om minnet har samma värde som rapporterats
// uppdateras inte minnet och någon sorts flagga skickas för att säga att uppdateringen inte gick att genomföra.

void
test_teardown()
{
  // Do not forget to free your stacks after each test
  // to avoid memory leaks
  free(stack);
}

void
test_finalize()
{
  // Destroy properly your test batch
}

int
test_push_safe()
{
  // Make sure your stack remains in a good state with expected content when
  // several threads push concurrently to it

  // Do some work
  stack_push(stack, &mutex, 1);
  int tmpValue = stack_pop(stack, &mutex);
  node_t* recycleNode = stack->pool.head->next;

  // Test node pool behaviour
  int simpleStackGotNode = assert(stack->pool.head->next != NULL);
  stack_push(stack, &mutex, tmpValue);
  int simpleStackRemovedNode = assert(stack->pool.head->next == NULL);
  int reusedNode = assert(stack->head->next == recycleNode);

  // Test shared stack behaviour
  stack_push(stack, &mutex, 2);
  stack_push(stack, &mutex, 3);
  stack_push(stack, &mutex, 4);

  // check if the stack is in a consistent state
  int res = assert(stack_check(stack));

  // check other properties expected after a push operation
  // (this is to be updated as your stack design progresses)
  // Now, the test succeeds
  return res && simpleStackGotNode && simpleStackRemovedNode && reusedNode && assert(
    stack->head->next->next->next->next->data == 1 &&
    stack->head->next->next->next->data == 2 &&
    stack->head->next->next->data == 3 &&
    stack->head->next->data == 4
    );
}

int
test_pop_safe()
{
  // Same as the test above for parallel pop operation
  stack_push(stack, &mutex, 1);
  stack_push(stack, &mutex, 2);
  stack_push(stack, &mutex, 3);
  stack_push(stack, &mutex, 4);
  
  return assert(
    stack_pop(stack, &mutex) == 4 &&
    stack_pop(stack, &mutex) == 3 &&
    stack_pop(stack, &mutex) == 2 &&
    stack_pop(stack, &mutex) == 1
  );
}

// 3 Threads should be enough to raise and detect the ABA problem
#define ABA_NB_THREADS 3

#if NON_BLOCKING == 1 || NON_BLOCKING == 2
pthread_mutex_t slow_pop_mutex;
pthread_mutex_t push_pool_mutex;

void init_slow_pop() {
  ABA_slow_pop(stack, &slow_pop_mutex);
}

void init_safe_pop() {
  printf("Thread 2 popping\n");
  stack_pop(stack, &mutex);
}

void init_safe_push() {
  printf("Thread one pushing\n");
  stack_push(stack, &mutex, 5);
}

void init_pool_wait_pop() {
  printf("Thread one popping\n");
  pool_wait_pop(stack, &push_pool_mutex);
}
#endif

int test_aba()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  //int success, aba_detected = 0;
  // Write here a test for the ABA problem

  pthread_mutex_init(&slow_pop_mutex, NULL);  
  pthread_mutex_init(&push_pool_mutex, NULL);

  // Init stack with 3 -> 2 -> 1
  stack_push(stack, &mutex, 1);
  stack_push(stack, &mutex, 2);
  stack_push(stack, &mutex, 3);

  pthread_attr_t attr;
  pthread_attr_init(&attr);

  pthread_t thread[ABA_NB_THREADS];

  pthread_mutex_lock(&slow_pop_mutex); // Make sure thread zero cannot continue its pop after saving the pointers
  pthread_mutex_lock(&push_pool_mutex); // Make sure thread one cannot push to the pool before thread two
    pthread_create(&thread[0], &attr, &init_slow_pop, NULL); // Thread zero starts

    pthread_create(&thread[1], &attr, &init_pool_wait_pop, NULL); // Thread one completes pop of A
    pthread_create(&thread[2], &attr, &init_safe_pop, NULL); // Thread two completes pop of B

    pthread_join(thread[2], NULL);
    pthread_mutex_unlock(&push_pool_mutex); // Allow thread one to push to pool

    pthread_join(thread[1], NULL);
    pthread_create(&thread[1], &attr, &init_safe_push, NULL); // Thread one completes push of A

    pthread_join(thread[1], NULL);
  pthread_mutex_unlock(&slow_pop_mutex); // Allow thread zero to finish pop
  pthread_join(thread[0], NULL);

  printf("\"Stack\" contents:\n");
  for(node_t* itr = stack->head->next; itr != NULL; itr = itr->next) {
    printf("  %d\n", itr->data);
  }

  if(stack->head->next->data == 2) {
    return 1;
  } else {
    return 0;
  }

  //success = aba_detected;
  //return success;
#else
  // No ABA is possible with lock-based synchronization. Let the test succeed only
  return 1;
#endif
}

// We test here the CAS function
struct thread_test_cas_args
{
  int id;
  size_t* counter;
  pthread_mutex_t *lock;
};
typedef struct thread_test_cas_args thread_test_cas_args_t;

void*
thread_test_cas(void* arg)
{
#if NON_BLOCKING != 0
  thread_test_cas_args_t *args = (thread_test_cas_args_t*) arg;
  int i;
  size_t old, local;

  for (i = 0; i < MAX_PUSH_POP; i++)
    {
      do {
        old = *args->counter;
        local = old + 1;
#if NON_BLOCKING == 1
      } while (cas(args->counter, old, local) != old);
#elif NON_BLOCKING == 2
      } while (software_cas(args->counter, old, local, args->lock) != old);
#endif
    }
#endif

  return NULL;
}

// Make sure Compare-and-swap works as expected
int
test_cas()
{
#if NON_BLOCKING == 1 || NON_BLOCKING == 2
  pthread_attr_t attr;
  pthread_t thread[NB_THREADS];
  thread_test_cas_args_t args[NB_THREADS];
  pthread_mutexattr_t mutex_attr;
  pthread_mutex_t lock;

  size_t counter;

  int i, success;

  counter = 0;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE); 
  pthread_mutexattr_init(&mutex_attr);
  pthread_mutex_init(&lock, &mutex_attr);

  for (i = 0; i < NB_THREADS; i++)
    {
      args[i].id = i;
      args[i].counter = &counter;
      args[i].lock = &lock;
      pthread_create(&thread[i], &attr, &thread_test_cas, (void*) &args[i]);
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }

  success = assert(counter == (size_t)(NB_THREADS * MAX_PUSH_POP));

  if (!success)
    {
      printf("Got %ti, expected %i. ", counter, NB_THREADS * MAX_PUSH_POP);
    }

  return success;
#else
  return 1;
#endif
}

int
main(int argc, char **argv)
{
setbuf(stdout, NULL);
// MEASURE == 0 -> run unit tests
#if MEASURE == 0
  test_init();

  test_run(test_cas);

  test_run(test_push_safe);
  test_run(test_pop_safe);
  test_run(test_aba);

  test_finalize();
#else
  int i;
  pthread_t thread[NB_THREADS];
  pthread_attr_t attr;
  stack_measure_arg_t arg[NB_THREADS];
  pthread_attr_init(&attr);

  test_init();
  test_setup();

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (i = 0; i < NB_THREADS; i++)
    {
      arg[i].id = i;
#if MEASURE == 1
      pthread_create(&thread[i], &attr, stack_measure_pop, (void*)&arg[i]);
#else
      pthread_create(&thread[i], &attr, stack_measure_push, (void*)&arg[i]);
#endif
    }

  for (i = 0; i < NB_THREADS; i++)
    {
      pthread_join(thread[i], NULL);
    }
  clock_gettime(CLOCK_MONOTONIC, &stop);

  // Print out results
  for (i = 0; i < NB_THREADS; i++)
    {
        printf("Thread %d time: %f\n", i, timediff(&t_start[i], &t_stop[i]));
    }
#endif

  return 0;
}
