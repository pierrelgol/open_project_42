/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   std.h                                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: pollivie <pollivie.student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/07/06 12:00:00 by pollivie          #+#    #+#             */
/*   Updated: 2024/07/06 12:00:00 by pollivie         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef STD_H
#define STD_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include "types.h"

#ifndef DEBUG
#define DEBUG 0
#endif

#ifndef VECTOR_DEFAULT_CAPACITY
#define VECTOR_DEFAULT_CAPACITY 32
#endif

#define ALPHABET_SIZE 26

/**********/
/* STRING */
/**********/

usize string_length(const char *const source);
usize string_copy(char *const dest, const char *const source, const usize dsize);
usize string_concat(char *const dest, const char *const source, const usize dsize);
usize string_span(const char *const source, const char *const delimiters);
usize string_count(const char *const source, const char *const delimiters);
usize string_wcount(const char *const source, const char *const delimiters);
usize string_cspan(const char *const source, const char *const delimiters);
isize string_index_of(const char *source, const i32 ch);
isize string_compare(const char *const source1, const char *const source2);
isize string_ncompare(const char *const source1, const char *const source2, const usize n);
char *string_clone(const char *const source);
char *string_nclone(const char *const source, const usize n);
char *string_join(const char *const source1, const char *const source2);
char *string_search(const char *const haystack, const char *const needle, const usize n);

/**********/
/* MEMORY */
/**********/

isize memory_compare(const void *const source1, const void *const source2, const usize n);
void *memory_search(const void *const source, const i32 byte, const usize ssize);
void *memory_ccopy(void *const dest, const i32 byte, const void *const source, const usize dsize);
void *memory_copy(void *const dest, const void *const source, const usize bytes);
void *memory_move(void *const dest, const void *const src, const usize bytes);
void *memory_fill(void *const dest, const i32 byte, const usize dsize);
void *memory_zero(void *const dest, const usize dsize);
void *memory_alloc(const usize size);
void *memory_dupe(const void *const source, const usize size);
void *memory_dupz(const void *const source);
void  memory_dealloc(void *const ptr);

/*********/
/* ASCII */
/*********/

bool is_alpha(const i32 ch);
bool is_alnum(const i32 ch);
bool is_space(const i32 ch);
bool is_ctrl(const i32 ch);
bool is_ascii(const i32 ch);
bool is_print(const i32 ch);
bool is_newline(const i32 ch);
bool is_punct(const i32 ch);
bool is_digit(const i32 ch);
bool is_lower(const i32 ch);
bool is_upper(const i32 ch);
bool is_hex(const i32 ch);
bool is_graph(const i32 ch);

/**********/
/* VECTOR */
/**********/

typedef struct s_vector {
        usize      is_sorted;
        usize      capacity;
        usize      count;
        usize      index;
        usize      saved;
        uintptr_t *data;
} t_vector;

t_vector *vector_create_with_capacity(const usize capacity);
t_vector *vector_create(void);
bool      vector_resize(t_vector *self, const usize new_capacity);
void      vector_clear(t_vector *self);
t_vector *vector_destroy(t_vector *self);
bool      vector_insert_front(t_vector *self, uintptr_t elem);
bool      vector_insert_back(t_vector *self, uintptr_t elem);
bool      vector_insert_after(t_vector *self, uintptr_t elem, const usize index);
bool      vector_insert_at(t_vector *self, uintptr_t elem, const usize index);
uintptr_t vector_remove_front(t_vector *self);
uintptr_t vector_remove_back(t_vector *self);
uintptr_t vector_remove_after(t_vector *self, const usize index);
uintptr_t vector_remove_at(t_vector *self, const usize index);
bool      vector_push(t_vector *vector, uintptr_t elem);
uintptr_t vector_pop(t_vector *vector);
bool      vector_enqueue(t_vector *vector, uintptr_t elem);
uintptr_t vector_dequeue(t_vector *vector);
uintptr_t vector_get_at(t_vector *vector, const usize index);
void      vector_set_at(t_vector *vector, uintptr_t elem, const usize index);
bool      vector_is_full(t_vector *self);
bool      vector_is_empty(t_vector *self);
void      vector_sort(t_vector *self, isize(cmp)(uintptr_t a, uintptr_t b));
isize     vector_search(t_vector *self, uintptr_t elem, isize(cmp)(uintptr_t a, uintptr_t b));
bool      vector_insert_sorted(t_vector *self, uintptr_t elem, isize(cmp)(uintptr_t a, uintptr_t b));

/* Vector Iterator Functions */
void      it_restore(t_vector *self);
void      it_advance(t_vector *self);
bool      it_contains(t_vector *self, uintptr_t elem, bool(eql)(uintptr_t a, uintptr_t b));
uintptr_t it_match(t_vector *self, uintptr_t elem, bool(eql)(uintptr_t a, uintptr_t b));
usize     it_skip(t_vector *self, uintptr_t elem, bool(eql)(uintptr_t a, uintptr_t b));
uintptr_t it_peek_next(t_vector *self);
uintptr_t it_peek_curr(t_vector *self);
uintptr_t it_peek_prev(t_vector *self);
void      it_save(t_vector *self);
bool      it_end(t_vector *self);

/*********/
/* LIST  */
/*********/

typedef struct node_t {
        struct node_t *next;
        uintptr_t      data;
} t_node;

t_node *node_create(uintptr_t data);
t_node *node_get_nchild(t_node *self, usize n);
void    node_insert_child(t_node *self, t_node *child);
t_node *node_remove_child(t_node *self);
usize   node_count_child(t_node *self);
t_node *node_next(t_node *self);
t_node *node_destroy(t_node *self);

typedef struct list_t {
        t_node *head;
        t_node *tail;
        usize   size;
} t_list;

t_list   *list_create(void);
void      list_insert_front(t_list *self, t_node *new_head);
void      list_insert_at(t_list *self, t_node *node, usize index);
void      list_insert_back(t_list *self, t_node *new_tail);
bool      list_is_empty(t_list *self);
usize     list_size(t_list *self);
t_node   *list_remove_front(t_list *self);
t_node   *list_remove_back(t_list *self);
t_node   *list_remove_at(t_list *self, usize index);
void      list_push_front(t_list *self, uintptr_t data);
void      list_push_back(t_list *self, uintptr_t data);
void      list_push_at(t_list *self, uintptr_t data, usize index);
uintptr_t list_pop_front(t_list *self);
uintptr_t list_pop_back(t_list *self);
uintptr_t list_pop_at(t_list *self, usize index);
void      list_sort(t_node **list, int (*f)(uintptr_t d1, uintptr_t d2));
t_list   *list_destroy(t_list *self);

/***********/
/* HASHMAP */
/***********/

typedef struct s_entry {
        char     *key;
        uintptr_t value;
} t_entry;

typedef struct s_hashmap {
        usize    size;
        t_entry *body;
        usize    capacity;
} t_hashmap;

t_hashmap *hashmap_create(usize capacity);
void       hashmap_destroy(t_hashmap *self);
void       hashmap_put(t_hashmap *self, char *key, uintptr_t value);
uintptr_t  hashmap_get(t_hashmap *self, char *key);
usize      hashmap_hash(char *str);
t_entry   *hashmap_body_create(usize capacity);
void       hashmap_body_remove(t_hashmap *self, char *key);
void       hashmap_body_resize(t_hashmap *self, usize capacity);
usize      hashmap_body_find_empty(t_hashmap *self, char *key);

bool       is_prime(usize num);
usize      find_next_prime(usize num);

/*********/
/* TRIE  */
/*********/

typedef struct s_trie_node {
        struct s_trie_node *children[ALPHABET_SIZE];
        bool                is_end_of_word;
} t_trie_node;

typedef struct Trie {
        t_trie_node *root;
} t_trie;

t_trie_node *trie_node_create(void);
t_trie_node *trie_node_find_prefix_node(t_trie_node *const self, const char *prefix);
void         trie_node_destroy(t_trie_node *self);
bool         trie_node_remove_child(t_trie_node *self, const char *const key, const usize depth);
bool         trie_node_is_empty(t_trie_node *self);

t_trie      *trie_create(void);
void         trie_insert(t_trie *const self, const char *const key);
bool         trie_search(t_trie *const self, const char *const key);
bool         trie_remove(t_trie *const self, const char *const key);
t_list      *trie_suggest(t_trie *const self, const char *prefix);
void         trie_collect_suggestions(t_trie_node *const node, const char *prefix, t_list *suggestions);
t_trie      *trie_destroy(t_trie *const self);

#endif /* STD_H */
