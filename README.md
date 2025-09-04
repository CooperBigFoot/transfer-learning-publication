# To Dos

## Implement data cleaning API

### Implement "atmoic" cleaners (=> the cleaning steps)

1. trim_nan (removes leading and trailing NaN and nulls for polars) DONE
2. clip (clips value above or below threshold to a specific number like clip(min=0, max=100)) DONE
3. fill_nan (takes a value, and a add_flag parameter. The flag paramet adds a {column_name}_flag which is True for values that have been filled) DONE
4. Split (implement only proportional split for now, leaving door open for other strategies)
5. Cyclical time encoding idk what to call it yet
