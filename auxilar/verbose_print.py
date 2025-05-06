def verbose_print_factory (verbose):
    return print if verbose else lambda x: None

if __name__ == "__main__":
    vprint = verbose_print_factory(True)
    vprint("Esto se tiene que ver")

    vprint = verbose_print_factory(False)
    vprint("Esto NO se tiene que ver")