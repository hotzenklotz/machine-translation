from collections import defaultdict


def grow_diag_final(len_e, len_f,  e2f, f2e):
    neighboring = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    alignment = set(e2f).intersection(set(f2e))
    union = set(e2f).union(set(f2e))

    aligned = defaultdict(set)
    for i, j in alignment:
        aligned["e"].add(i)
        aligned["j"].add(j)

    def grow_diag():

        prev_len = len(alignment) - 1
        # iterate until no new points added
        while prev_len < len(alignment):
            # for english word e = 0 ... en
            for e in range(len_e):
                # for foreign word f = 0 ... fn
                for f in range(len_f):
                    # if ( e aligned with f)
                    if (e, f) in alignment:
                        # for each neighboring point (e-new, f-new)
                        for neighbor in neighboring:
                            e_new = e + neighbor[0]
                            f_new = f + neighbor[1]
                            # if ( ( e-new not aligned and f-new not aligned)
                            # and (e-new, f-new in union(e2f, f2e) )
                            if (e_new not in aligned and f_new not in aligned) and neighbor in union:
                                alignment.add(neighbor)
                                aligned['e'].add(e_new)
                                aligned['f'].add(f_new)
                                prev_len += 1

    def final(a):
        # for english word e = 0 ... en
        for e_new in range(len_e):
            # for foreign word f = 0 ... fn
            for f_new in range(len_f):
                # if ( ( e-new not aligned and f-new not aligned)
                # and (e-new, f-new in union(e2f, f2e) )
                if e_new not in aligned and f_new not in aligned and (e_new, f_new) in a:
                    alignment.add((e_new, f_new))
                    aligned['e'].add(e_new)
                    aligned['f'].add(f_new)

    grow_diag()
    final(e2f)
    final(f2e)

    return alignment
