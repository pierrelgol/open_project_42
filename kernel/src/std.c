/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   std.c                                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: pollivie <pollivie.student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2024/07/06 12:00:00 by pollivie          #+#    #+#             */
/*   Updated: 2024/07/06 12:00:00 by pollivie         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include "std.h"

inline bool is_alnum(const i32 ch) {
        return (((ch | 32) >= 'a' && (ch | 32) <= 'z') || (ch >= '0' && ch <= '9'));
}

inline bool is_alpha(const i32 ch) {
        return ((ch | 32) >= 'a' && (ch | 32) <= 'z');
}

inline bool is_ascii(const i32 ch) {
        return (ch >= 0 && ch <= 127);
}

inline bool is_ctrl(const i32 ch) {
        return (ch == 127 || (ch >= 0 && ch <= 31));
}

inline bool is_digit(const i32 ch) {
        return (ch >= '0' && ch <= '9');
}

inline bool is_graph(const i32 ch) {
        return (!is_ctrl(ch) && !is_space(ch));
}

inline bool is_hex(const i32 ch) {
        return ((ch >= '0' && ch <= '9') || ((ch | 32) >= 'a' && (ch | 32) <= 'f'));
}

inline bool is_lower(const i32 ch) {
        return (ch >= 'a' && ch <= 'z');
}

inline bool is_newline(const i32 ch) {
        return (ch == '\n');
}

inline bool is_print(const i32 ch) {
        return (ch >= ' ' && ch <= '~');
}

inline bool is_punct(const i32 ch) {
        return (ch != 32 && !is_alnum(ch) && !is_ctrl(ch));
}

inline bool is_space(const i32 ch) {
        return (ch == ' ' || (ch >= 9 && ch <= 13));
}

inline bool is_upper(const i32 ch) {
        return (ch >= 'A' && ch <= 'Z');
}


void *memory_alloc(const usize size) {
        static u8    heap[8192];
        static usize heap_ptr = 0;
        u8          *ptr;

        if (heap_ptr + size > sizeof(heap)) return (NULL);

        ptr = &heap[heap_ptr];
        heap_ptr += size;
        return (memory_zero(ptr, size));
}

void *memory_ccopy(void *const dest, const i32 byte, const void *const source, const usize dsize) {
        u8 *sptr;
        u8 *dptr;
        u8  b;
        u8  i;

        dptr = (u8 *)dest;
        sptr = (u8 *)source;
        b    = (u8)byte;
        i    = 0;
        while (i < dsize) {
                if (*(sptr + i) == b) return ((void *)(source + i + 1));
                *(dptr + i) = *(sptr + i);
                ++i;
        }
        return (NULL);
}

isize memory_compare(const void *const source1, const void *const source2, const usize n) {
        const u8 *s1;
        const u8 *s2;
        usize     i;

        s1 = (const u8 *)source1;
        s2 = (const u8 *)source2;
        i  = 0;
        while (i < n) {
                if (*(s1 + i) != *(s2 + i)) return (*(s1 + i) - *(s2 + i));
                ++i;
        }
        return (0);
}

void *memory_copy(void *const dest, const void *const source, const usize bytes) {
        u8       *dptr;
        const u8 *sptr;
        usize     i;

        dptr = (u8 *)dest;
        sptr = (const u8 *)source;
        i    = 0;
        while (i < bytes) {
                *(dptr + i) = *(sptr + i);
                ++i;
        }
        return (dest);
}

void *memory_dupe(const void *const source, const usize size) {
        void *duplicate;

        duplicate = memory_alloc(size);
        if (!duplicate) return (NULL);
        return (memory_copy(duplicate, source, size));
}

void *memory_dupz(const void *const source) {
        return (memory_dupe(source, string_length((char *)source) + 1));
}

void memory_dealloc(void *const ptr) {
        (void)ptr;
}

void *memory_fill(void *const dest, const i32 byte, const usize dsize) {
        u8   *dptr;
        u8    b;
        usize i;

        dptr = (u8 *)dest;
        b    = (u8)byte;
        i    = 0;
        while (i < dsize) {
                *(dptr + i) = b;
                ++i;
        }
        return (dest);
}

void *memory_move(void *const dest, const void *const src, const usize bytes) {
        u8       *dptr;
        const u8 *sptr;
        usize     i;

        dptr = (u8 *)dest;
        sptr = (const u8 *)src;
        if (dptr < sptr) {
                i = 0;
                while (i < bytes) {
                        *(dptr + i) = *(sptr + i);
                        ++i;
                }
        } else {
                i = bytes;
                while (i > 0) {
                        *(dptr + i - 1) = *(sptr + i - 1);
                        --i;
                }
        }
        return (dest);
}

void *memory_search(const void *const source, const i32 byte, const usize ssize) {
        const u8 *sptr;
        u8        b;
        usize     i;

        sptr = (const u8 *)source;
        b    = (u8)byte;
        i    = 0;
        while (i < ssize) {
                if (*(sptr + i) == b) return ((void *)(source + i));
                ++i;
        }
        return (NULL);
}

void *memory_zero(void *const dest, const usize dsize) {
        return (memory_fill(dest, 0x00, dsize));
}

/**********/
/* STRING */
/**********/

usize string_cspan(const char *const source, const char *const delimiters) {
        usize i;
        usize j;

        i = 0;
        while (*(source + i)) {
                j = 0;
                while (*(delimiters + j)) {
                        if (*(source + i) == *(delimiters + j)) return (i);
                        ++j;
                }
                ++i;
        }
        return (i);
}

usize string_span(const char *const source, const char *const delimiters) {
        usize i;
        usize j;
        bool  found;

        i = 0;
        while (*(source + i)) {
                found = false;
                j     = 0;
                while (*(delimiters + j)) {
                        if (*(source + i) == *(delimiters + j)) {
                                found = true;
                                break;
                        }
                        ++j;
                }
                if (!found) return (i);
                ++i;
        }
        return (i);
}

usize string_count(const char *const source, const char *const delimiters) {
        usize count;
        usize i;
        usize j;

        count = 0;
        i     = 0;
        while (*(source + i)) {
                j = 0;
                while (*(delimiters + j)) {
                        if (*(source + i) == *(delimiters + j)) {
                                ++count;
                                break;
                        }
                        ++j;
                }
                ++i;
        }
        return (count);
}

usize string_wcount(const char *const source, const char *const delimiters) {
        usize count;
        usize i;
        bool  in_word;

        count   = 0;
        i       = 0;
        in_word = false;
        while (*(source + i)) {
                if (string_span(source + i, delimiters) > 0) {
                        if (!in_word) {
                                ++count;
                                in_word = true;
                        }
                        i += string_span(source + i, delimiters);
                } else {
                        in_word = false;
                        ++i;
                }
        }
        return (count);
}

isize string_compare(const char *const source1, const char *const source2) {
        usize i;

        i = 0;
        while (*(source1 + i) && *(source2 + i)) {
                if (*(source1 + i) != *(source2 + i)) return (*(source1 + i) - *(source2 + i));
                ++i;
        }
        return (*(source1 + i) - *(source2 + i));
}

isize string_ncompare(const char *const source1, const char *const source2, const usize n) {
        usize i;

        i = 0;
        while (i < n && *(source1 + i) && *(source2 + i)) {
                if (*(source1 + i) != *(source2 + i)) return (*(source1 + i) - *(source2 + i));
                ++i;
        }
        if (i == n) return (0);
        return (*(source1 + i) - *(source2 + i));
}

usize string_copy(char *const dest, const char *const source, const usize dsize) {
        usize i;

        i = 0;
        while (i < dsize - 1 && *(source + i)) {
                *(dest + i) = *(source + i);
                ++i;
        }
        *(dest + i) = '\0';
        return (i);
}

usize string_concat(char *const dest, const char *const source, const usize dsize) {
        usize dest_len;
        usize i;

        dest_len = string_length(dest);
        i        = 0;
        while (dest_len + i < dsize - 1 && *(source + i)) {
                *(dest + dest_len + i) = *(source + i);
                ++i;
        }
        *(dest + dest_len + i) = '\0';
        return (dest_len + i);
}

char *string_clone(const char *const source) {
        usize len;
        char *clone;

        len   = string_length(source);
        clone = memory_alloc(len + 1);
        if (!clone) return (NULL);
        string_copy(clone, source, len + 1);
        return (clone);
}

char *string_nclone(const char *const source, const usize n) {
        char *clone;

        clone = memory_alloc(n + 1);
        if (!clone) return (NULL);
        string_copy(clone, source, n + 1);
        return (clone);
}

isize string_index_of(const char *source, const i32 ch) {
        usize i;

        i = 0;
        while (*(source + i)) {
                if (*(source + i) == ch) return (i);
                ++i;
        }
        return (-1);
}

char *string_join(const char *const source1, const char *const source2) {
        usize len1;
        usize len2;
        char *result;

        len1   = string_length(source1);
        len2   = string_length(source2);
        result = memory_alloc(len1 + len2 + 1);
        if (!result) return (NULL);
        string_copy(result, source1, len1 + 1);
        string_concat(result, source2, len1 + len2 + 1);
        return (result);
}

usize string_length(const char *const source) {
        usize len;

        len = 0;
        while (*(source + len)) ++len;
        return (len);
}

char *string_search(const char *const haystack, const char *const needle, const usize n) {
        usize i;
        usize j;
        usize needle_len;

        needle_len = string_length(needle);
        if (needle_len == 0) return ((char *)haystack);
        if (needle_len > n) return (NULL);
        i = 0;
        while (i <= n - needle_len) {
                j = 0;
                while (j < needle_len && *(haystack + i + j) == *(needle + j)) ++j;
                if (j == needle_len) return ((char *)(haystack + i));
                ++i;
        }
        return (NULL);
}

/**********/
/* VECTOR */
/**********/

t_vector *vector_create(void) {
        t_vector *self;

        self = (t_vector *)memory_alloc(sizeof(t_vector));
        if (!self) return (NULL);
        self->capacity  = VECTOR_DEFAULT_CAPACITY;
        self->count     = 0;
        self->is_sorted = false;
        self->data      = memory_alloc(VECTOR_DEFAULT_CAPACITY * sizeof(uintptr_t));
        if (!self->data) return (vector_destroy(self));
        return (self);
}

t_vector *vector_create_with_capacity(const usize capacity) {
        t_vector *self;

        self = (t_vector *)memory_alloc(sizeof(t_vector));
        if (!self) return (NULL);
        self->capacity = capacity;
        self->count    = 0;
        self->data     = memory_alloc(capacity * sizeof(uintptr_t));
        if (!self->data) return (vector_destroy(self));
        return (self);
}

void vector_clear(t_vector *self) {
        memory_fill(self->data, 0x00, self->count * sizeof(uintptr_t));
        self->count = 0;
        self->index = 0;
        self->saved = 0;
}

t_vector *vector_destroy(t_vector *self) {
        if (self) {
                if (self->data) memory_dealloc(self->data);
                memory_dealloc(self);
        }
        return (NULL);
}

uintptr_t vector_get_at(t_vector *vector, const usize index) {
        if (vector_is_empty(vector) || index > vector->count) return (0);
        return (vector->data[index]);
}

void vector_set_at(t_vector *vector, uintptr_t elem, const usize index) {
        if (vector_is_empty(vector) || index > vector->count) return;
        vector->data[index] = elem;
}

bool vector_push(t_vector *self, uintptr_t elem) {
        return (vector_insert_at(self, elem, self->count));
}

uintptr_t vector_pop(t_vector *self) {
        if (vector_is_empty(self)) return (0);
        return (vector_remove_at(self, self->count - 1));
}

bool vector_enqueue(t_vector *self, uintptr_t elem) {
        return (vector_insert_at(self, elem, self->count));
}

uintptr_t vector_dequeue(t_vector *self) {
        if (vector_is_empty(self)) return (0);
        return (vector_remove_at(self, 0));
}

bool vector_is_empty(t_vector *self) {
        return (self->count == 0);
}

bool vector_is_full(t_vector *self) {
        return (self->count == self->capacity);
}

bool vector_insert_front(t_vector *self, uintptr_t elem) {
        return (vector_insert_at(self, elem, 0));
}

bool vector_insert_back(t_vector *self, uintptr_t elem) {
        return (vector_insert_at(self, elem, self->count));
}

bool vector_insert_after(t_vector *self, uintptr_t elem, const usize index) {
        return (vector_insert_at(self, elem, index + 1));
}

bool vector_insert_at(t_vector *self, uintptr_t elem, const usize index) {
        usize bytes_to_move;

        if (index > self->count) return (false);
        if (vector_is_full(self)) {
                if (!vector_resize(self, self->capacity * 2)) return (false);
        }
        bytes_to_move = (self->count - index) * sizeof(uintptr_t);
        memory_move((self->data + index + 1), (self->data + index), bytes_to_move);
        self->data[index] = elem;
        self->count += 1;
        return (true);
}

bool vector_insert_sorted(t_vector *self, uintptr_t elem, isize(cmp)(uintptr_t a, uintptr_t b)) {
        usize pos;
        usize i;

        if (self == NULL || self->data == NULL) return (false);
        pos = 0;
        if (self->is_sorted == 0) vector_sort(self, cmp);
        if (self->count == self->capacity) {
                if (!vector_resize(self, self->capacity * 2)) return (false);
        }
        while (pos < self->count && cmp(self->data[pos], elem) < 0) pos++;
        i = self->count;
        while (i > pos) {
                self->data[i] = self->data[i - 1];
                i -= 1;
        }
        self->data[pos] = elem;
        self->count++;
        return (true);
}

uintptr_t vector_remove_front(t_vector *self) {
        if (vector_is_empty(self)) return (0);
        return (vector_remove_at(self, 0));
}

uintptr_t vector_remove_back(t_vector *self) {
        if (vector_is_empty(self)) return (0);
        return (vector_remove_at(self, self->count - 1));
}

uintptr_t vector_remove_after(t_vector *self, const usize index) {
        if (vector_is_empty(self)) return (0);
        return (vector_remove_at(self, index + 1));
}

uintptr_t vector_remove_at(t_vector *self, const usize index) {
        usize     bytes_to_move;
        uintptr_t elem;

        if (vector_is_empty(self) || index >= self->count) return (0);
        elem          = self->data[index];
        bytes_to_move = (self->count - index - 1) * sizeof(uintptr_t);
        if (bytes_to_move) memory_move(self->data + index, self->data + index + 1, bytes_to_move);
        self->count -= 1;
        return (elem);
}

bool vector_resize(t_vector *self, const usize new_capacity) {
        uintptr_t *old_data;
        usize      old_capacity;

        old_capacity = self->count;
        old_data     = self->data;
        self->data   = (uintptr_t *)memory_alloc(new_capacity * sizeof(uintptr_t));
        if (!self->data) {
                self->data = old_data;
                return (false);
        }
        memory_copy(self->data, old_data, old_capacity * sizeof(uintptr_t));
        self->capacity = new_capacity;
        memory_dealloc(old_data);
        return (true);
}

static isize binary_search(t_vector *self, uintptr_t elem, isize(cmp)(uintptr_t a, uintptr_t b)) {
        isize left;
        isize mid;
        isize right;
        isize comparison;

        left  = 0;
        mid   = 0;
        right = self->count - 1;
        while (left <= right) {
                mid        = left + (right - left) / 2;
                comparison = cmp(self->data[mid], elem);
                if (comparison == 0)
                        return (mid);
                else if (comparison < 0)
                        left = mid + 1;
                else
                        right = mid - 1;
        }
        return (-1);
}

static isize linear_search(t_vector *self, uintptr_t elem, isize(cmp)(uintptr_t a, uintptr_t b)) {
        usize i;

        i = 0;
        while (i < self->count) {
                if (cmp(self->data[i], elem) == 0) return (i);
                i += 1;
        }
        return (-1);
}

isize vector_search(t_vector *self, uintptr_t elem, isize(cmp)(uintptr_t a, uintptr_t b)) {
        if (self == NULL || self->data == NULL) return (-1);
        if (self->is_sorted)
                return (binary_search(self, elem, cmp));
        else
                return (linear_search(self, elem, cmp));
        return (-1);
}

static void swap(uintptr_t *const a, uintptr_t *const b) {
        const uintptr_t temp = *a;

        *a                   = *b;
        *b                   = temp;
}

static void quicksort(uintptr_t *data, isize left, isize right, isize(cmp)(uintptr_t a, uintptr_t b)) {
        isize pivot;
        isize i;
        isize j;

        if (left >= right) return;
        i     = left;
        j     = right;
        pivot = data[(left + right) / 2];
        while (i <= j) {
                while (cmp(data[i], pivot) < 0) i++;
                while (cmp(data[j], pivot) > 0) j--;
                if (i <= j) {
                        swap(&data[i], &data[j]);
                        i++;
                        j--;
                }
        }
        quicksort(data, left, j, cmp);
        quicksort(data, i, right, cmp);
}

void vector_sort(t_vector *self, isize(cmp)(uintptr_t a, uintptr_t b)) {
        if (self == NULL || self->data == NULL || self->count < 2) return;
        quicksort(self->data, 0, self->count - 1, cmp);
        self->is_sorted = 1;
}

bool it_end(t_vector *self) {
        return (self->index == self->count);
}

void it_save(t_vector *self) {
        self->saved = self->index;
}

void it_restore(t_vector *self) {
        self->index = self->saved;
}

void it_advance(t_vector *self) {
        if (it_end(self)) return;
        self->index += 1;
}

bool it_contains(t_vector *self, uintptr_t elem, bool(eql)(uintptr_t a, uintptr_t b)) {
        const usize saved = self->index;
        uintptr_t   maybe_elem;

        if (it_end(self)) return (false);
        while (!it_end(self)) {
                maybe_elem = it_peek_curr(self);
                if (eql(maybe_elem, elem)) {
                        self->index = saved;
                        return (true);
                }
                it_advance(self);
        }
        self->index = saved;
        return (false);
}

uintptr_t it_match(t_vector *self, uintptr_t elem, bool(eql)(uintptr_t a, uintptr_t b)) {
        uintptr_t maybe_elem;

        if (it_end(self)) return (0);
        maybe_elem = it_peek_next(self);
        if (!maybe_elem) return (0);
        if (eql(maybe_elem, elem)) return (elem);
        return (0);
}

usize it_skip(t_vector *self, uintptr_t elem, bool(eql)(uintptr_t a, uintptr_t b)) {
        usize count;

        count = 0;
        while (it_match(self, elem, eql) != 0) {
                it_advance(self);
                count += 1;
        }
        return (count);
}

uintptr_t it_peek_next(t_vector *self) {
        if (self->index + 1 >= self->count) return (0);
        return (*(self->data + self->index + 1));
}

uintptr_t it_peek_curr(t_vector *self) {
        if (self->index >= self->count) return (0);
        return (*(self->data + self->index));
}

uintptr_t it_peek_prev(t_vector *self) {
        if (self->index < 1 || self->count == 0) return (0);
        return (*(self->data + self->index - 1));
}

/*********/
/* LIST  */
/*********/

t_node *node_create(uintptr_t data) {
        t_node *node;

        node = (t_node *)memory_alloc(sizeof(t_node));
        if (!node) return (NULL);
        node->data = data;
        node->next = NULL;
        return (node);
}

t_node *node_get_nchild(t_node *self, usize n) {
        usize i;

        i = 0;
        while (i < n && self) {
                self = self->next;
                i++;
        }
        return (self);
}

void node_insert_child(t_node *self, t_node *child) {
        if (!self || !child) return;
        child->next = self->next;
        self->next  = child;
}

t_node *node_remove_child(t_node *self) {
        t_node *child;

        if (!self || !self->next) return (NULL);
        child       = self->next;
        self->next  = child->next;
        child->next = NULL;
        return (child);
}

usize node_count_child(t_node *self) {
        usize count;

        count = 0;
        while (self) {
                count++;
                self = self->next;
        }
        return (count);
}

t_node *node_next(t_node *self) {
        return (self->next);
}

t_node *node_destroy(t_node *self) {
        if (self) memory_dealloc(self);
        return (NULL);
}

t_list *list_create(void) {
        t_list *list;

        list       = (t_list *)memory_alloc(sizeof(t_list));
        list->head = NULL;
        list->tail = NULL;
        list->size = 0;
        return (list);
}

t_list *list_destroy(t_list *self) {
        t_node *node;

        if (!self) return (NULL);
        while (!list_is_empty(self)) {
                node = list_remove_front(self);
                node_destroy(node);
        }
        memory_dealloc(self);
        return (NULL);
}

void list_insert_front(t_list *self, t_node *new_head) {
        if (list_is_empty(self)) {
                self->head = new_head;
                self->tail = new_head;
        } else if (self->size == 1) {
                self->head     = new_head;
                new_head->next = self->tail;
        } else {
                new_head->next = self->head;
                self->head     = new_head;
        }
        self->size += 1;
}

void list_insert_back(t_list *self, t_node *new_tail) {
        if (list_is_empty(self)) {
                self->head = new_tail;
                self->tail = new_tail;
        } else if (self->size == 1) {
                self->head->next = new_tail;
                self->tail       = new_tail;
        } else {
                self->tail->next = new_tail;
                self->tail       = new_tail;
        }
        self->size += 1;
}

void list_insert_at(t_list *self, t_node *node, usize index) {
        t_node *temp;

        if (index == 0 || list_is_empty(self))
                list_insert_front(self, node);
        else if (index >= self->size)
                list_insert_back(self, node);
        else {
                temp = node_get_nchild(self->head, index - 1);
                node_insert_child(temp, node);
                self->size += 1;
        }
}

bool list_is_empty(t_list *self) {
        return (self->size == 0);
}

usize list_size(t_list *self) {
        return (self->size);
}

t_node *list_remove_front(t_list *self) {
        t_node *old_head;

        if (list_is_empty(self)) return (NULL);
        old_head = self->head;
        if (self->size == 1) {
                self->head = NULL;
                self->tail = NULL;
        } else
                self->head = self->head->next;
        self->size -= 1;
        old_head->next = NULL;
        return (old_head);
}

t_node *list_remove_back(t_list *self) {
        t_node *old_tail;
        t_node *current;

        if (list_is_empty(self)) return (NULL);
        old_tail = self->tail;
        if (self->size == 1) {
                self->head = NULL;
                self->tail = NULL;
        } else {
                current = self->head;
                while (current->next != self->tail) current = current->next;
                current->next = NULL;
                self->tail    = current;
        }
        self->size -= 1;
        return (old_tail);
}

t_node *list_remove_at(t_list *self, usize index) {
        t_node *node_to_remove;
        t_node *prev_node;

        if (list_is_empty(self) || index >= self->size) return (NULL);
        if (index == 0) return (list_remove_front(self));
        if (index == self->size - 1) return (list_remove_back(self));
        prev_node            = node_get_nchild(self->head, index - 1);
        node_to_remove       = prev_node->next;
        prev_node->next      = node_to_remove->next;
        node_to_remove->next = NULL;
        self->size -= 1;
        return (node_to_remove);
}

void list_push_front(t_list *self, uintptr_t data) {
        t_node *new_node;

        new_node = node_create(data);
        if (new_node) list_insert_front(self, new_node);
}

void list_push_back(t_list *self, uintptr_t data) {
        t_node *new_node;

        new_node = node_create(data);
        if (new_node) list_insert_back(self, new_node);
}

void list_push_at(t_list *self, uintptr_t data, usize index) {
        t_node *new_node;

        new_node = node_create(data);
        if (new_node) list_insert_at(self, new_node, index);
}

uintptr_t list_pop_front(t_list *self) {
        t_node   *node;
        uintptr_t data;

        node = list_remove_front(self);
        if (!node) return (0);
        data = node->data;
        node_destroy(node);
        return (data);
}

uintptr_t list_pop_back(t_list *self) {
        t_node   *node;
        uintptr_t data;

        node = list_remove_back(self);
        if (!node) return (0);
        data = node->data;
        node_destroy(node);
        return (data);
}

uintptr_t list_pop_at(t_list *self, usize index) {
        t_node   *node;
        uintptr_t data;

        node = list_remove_at(self, index);
        if (!node) return (0);
        data = node->data;
        node_destroy(node);
        return (data);
}

void list_sort(t_node **list, int (*f)(uintptr_t d1, uintptr_t d2)) {
        t_node   *current;
        t_node   *next;
        uintptr_t temp;

        if (!list || !*list) return;
        current = *list;
        while (current->next) {
                next = current->next;
                while (next) {
                        if (f(current->data, next->data) > 0) {
                                temp          = current->data;
                                current->data = next->data;
                                next->data    = temp;
                        }
                        next = next->next;
                }
                current = current->next;
        }
}

/***********/
/* HASHMAP */
/***********/

bool is_prime(usize num) {
        usize i;

        if (num < 2) return (false);
        if (num == 2) return (true);
        if (num % 2 == 0) return (false);
        i = 3;
        while (i * i <= num) {
                if (num % i == 0) return (false);
                i += 2;
        }
        return (true);
}

usize find_next_prime(usize num) {
        if (num <= 2) return (2);
        if (num % 2 == 0) num++;
        while (!is_prime(num)) num += 2;
        return (num);
}

usize hashmap_hash(char *str) {
        usize hash;
        usize i;

        hash = 5381;
        i    = 0;
        while (str[i]) {
                hash = ((hash << 5) + hash) + str[i];
                i++;
        }
        return (hash);
}

t_entry *hashmap_body_create(usize capacity) {
        t_entry *body;
        usize    i;

        body = (t_entry *)memory_alloc(capacity * sizeof(t_entry));
        if (!body) return (NULL);
        i = 0;
        while (i < capacity) {
                body[i].key   = NULL;
                body[i].value = 0;
                i++;
        }
        return (body);
}

usize hashmap_body_find_empty(t_hashmap *self, char *key) {
        usize hash;
        usize index;
        usize i;

        hash  = hashmap_hash(key);
        index = hash % self->capacity;
        i     = 0;
        while (i < self->capacity) {
                if (self->body[index].key == NULL) return (index);
                if (string_compare(self->body[index].key, key) == 0) return (index);
                index = (index + 1) % self->capacity;
                i++;
        }
        return (index);
}

void hashmap_body_resize(t_hashmap *self, usize capacity) {
        t_entry *old_body;
        usize    old_capacity;
        usize    i;

        old_body       = self->body;
        old_capacity   = self->capacity;
        self->capacity = capacity;
        self->body     = hashmap_body_create(capacity);
        self->size     = 0;
        i              = 0;
        while (i < old_capacity) {
                if (old_body[i].key != NULL) hashmap_put(self, old_body[i].key, old_body[i].value);
                i++;
        }
        memory_dealloc(old_body);
}

t_hashmap *hashmap_create(usize capacity) {
        t_hashmap *self;

        capacity = find_next_prime(capacity);
        self     = (t_hashmap *)memory_alloc(sizeof(t_hashmap));
        if (!self) return (NULL);
        self->size     = 0;
        self->capacity = capacity;
        self->body     = hashmap_body_create(capacity);
        if (!self->body) {
                memory_dealloc(self);
                return (NULL);
        }
        return (self);
}

void hashmap_destroy(t_hashmap *self) {
        usize i;

        if (!self) return;
        i = 0;
        while (i < self->capacity) {
                if (self->body[i].key) memory_dealloc(self->body[i].key);
                i++;
        }
        memory_dealloc(self->body);
        memory_dealloc(self);
}

void hashmap_put(t_hashmap *self, char *key, uintptr_t value) {
        usize index;
        char *key_copy;

        if (!self || !key) return;
        if (self->size >= self->capacity / 2) hashmap_body_resize(self, find_next_prime(self->capacity * 2));
        index = hashmap_body_find_empty(self, key);
        if (self->body[index].key == NULL) {
                key_copy = string_clone(key);
                if (!key_copy) return;
                self->body[index].key   = key_copy;
                self->body[index].value = value;
                self->size++;
        } else
                self->body[index].value = value;
}

uintptr_t hashmap_get(t_hashmap *self, char *key) {
        usize index;

        if (!self || !key) return (0);
        index = hashmap_body_find_empty(self, key);
        if (self->body[index].key == NULL) return (0);
        return (self->body[index].value);
}

void hashmap_body_remove(t_hashmap *self, char *key) {
        usize index;

        if (!self || !key) return;
        index = hashmap_body_find_empty(self, key);
        if (self->body[index].key != NULL) {
                memory_dealloc(self->body[index].key);
                self->body[index].key   = NULL;
                self->body[index].value = 0;
                self->size--;
        }
}

/*********/
/* TRIE  */
/*********/

t_trie_node *trie_node_create(void) {
        t_trie_node *node;
        usize        i;

        node = (t_trie_node *)memory_alloc(sizeof(t_trie_node));
        if (!node) return (NULL);
        node->is_end_of_word = false;
        i                    = 0;
        while (i < ALPHABET_SIZE) {
                node->children[i] = NULL;
                i++;
        }
        return (node);
}

t_trie_node *trie_node_find_prefix_node(t_trie_node *const self, const char *prefix) {
        t_trie_node *current;
        usize        i;

        if (!self || !prefix) return (NULL);
        current = self;
        i       = 0;
        while (prefix[i]) {
                if (prefix[i] < 'a' || prefix[i] > 'z') return (NULL);
                if (!current->children[prefix[i] - 'a']) return (NULL);
                current = current->children[prefix[i] - 'a'];
                i++;
        }
        return (current);
}

void trie_node_destroy(t_trie_node *self) {
        usize i;

        if (!self) return;
        i = 0;
        while (i < ALPHABET_SIZE) {
                if (self->children[i]) trie_node_destroy(self->children[i]);
                i++;
        }
        memory_dealloc(self);
}

bool trie_node_remove_child(t_trie_node *self, const char *const key, const usize depth) {
        usize index;

        if (!self) return (false);
        if (depth == string_length(key)) {
                if (self->is_end_of_word) {
                        self->is_end_of_word = false;
                        return (trie_node_is_empty(self));
                }
                return (false);
        }
        index = key[depth] - 'a';
        if (!self->children[index]) return (false);
        if (trie_node_remove_child(self->children[index], key, depth + 1)) {
                trie_node_destroy(self->children[index]);
                self->children[index] = NULL;
                return (trie_node_is_empty(self));
        }
        return (false);
}

bool trie_node_is_empty(t_trie_node *self) {
        usize i;

        if (!self) return (true);
        i = 0;
        while (i < ALPHABET_SIZE) {
                if (self->children[i]) return (false);
                i++;
        }
        return (true);
}

t_trie *trie_create(void) {
        t_trie *trie;

        trie = (t_trie *)memory_alloc(sizeof(t_trie));
        if (!trie) return (NULL);
        trie->root = trie_node_create();
        if (!trie->root) {
                memory_dealloc(trie);
                return (NULL);
        }
        return (trie);
}

void trie_insert(t_trie *const self, const char *const key) {
        t_trie_node *current;
        usize        i;
        usize        index;

        if (!self || !key) return;
        current = self->root;
        i       = 0;
        while (key[i]) {
                if (key[i] < 'a' || key[i] > 'z') return;
                index = key[i] - 'a';
                if (!current->children[index]) current->children[index] = trie_node_create();
                current = current->children[index];
                i++;
        }
        current->is_end_of_word = true;
}

bool trie_search(t_trie *const self, const char *const key) {
        t_trie_node *node;

        if (!self || !key) return (false);
        node = trie_node_find_prefix_node(self->root, key);
        return (node && node->is_end_of_word);
}

bool trie_remove(t_trie *const self, const char *const key) {
        if (!self || !key) return (false);
        return (trie_node_remove_child(self->root, key, 0));
}

static char *trie_build_prefix(const char *prefix, char new_char) {
        const usize prefix_len = string_length(prefix);
        const usize new_len    = prefix_len + 2;
        char       *new_prefix;
        usize       i;

        new_prefix = (char *)memory_alloc(new_len);
        if (new_prefix) {
                i = 0;
                while (i < prefix_len) {
                        new_prefix[i] = prefix[i];
                        ++i;
                }
                new_prefix[prefix_len]     = new_char;
                new_prefix[prefix_len + 1] = '\0';
        }
        return (new_prefix);
}

void trie_collect_suggestions(t_trie_node *const node, const char *prefix, t_list *suggestions) {
        char *prefix_buffer;
        char  new_char;
        usize i;

        if (!node) return;
        if (node->is_end_of_word) list_push_back(suggestions, (uintptr_t)string_clone(prefix));
        i = 0;
        while (i < ALPHABET_SIZE) {
                if (node->children[i]) {
                        new_char      = 'a' + i;
                        prefix_buffer = trie_build_prefix(prefix, new_char);
                        if (prefix_buffer) {
                                trie_collect_suggestions(node->children[i], prefix_buffer, suggestions);
                                memory_dealloc(prefix_buffer);
                        }
                }
                ++i;
        }
}

t_list *trie_suggest(t_trie *const self, const char *prefix) {
        t_trie_node *prefix_node;
        t_list      *suggestions;

        if (!self || !prefix) return (NULL);
        suggestions = list_create();
        if (!suggestions) return (NULL);
        prefix_node = trie_node_find_prefix_node(self->root, prefix);
        if (prefix_node) trie_collect_suggestions(prefix_node, prefix, suggestions);
        return (suggestions);
}

t_trie *trie_destroy(t_trie *const self) {
        if (!self) return (NULL);
        if (self->root) trie_node_destroy(self->root);
        memory_dealloc(self);
        return (NULL);
}
