import tensorflow as tf
import tensorflow.contrib.graph_editor as ge


def get_tentacles(scope_name, within_ops=None):
    """
    Retrieve the incoming gate (the holes) of the layer
    (a layer is identified by its name scope), the ten-
    tacles that goes into that hole; also, retrieve the
    outcoming tentacle of the layer, and the holes that
    these tentacles goes into.

    Args:
      scope_name: tf graph name scope; a layer is iden-
                  tified by the largest common scope of
                  its ops.
      within_ops: a list of ops within which the search
                  is restricted; if `within_ops` is set
                  to None, the search will be performed
                  on the whole `tf.default_graph()`.
    Returns:
      incoming:   a dictionary `{op: [tentacle_list]}`,
                  where `op` is the layers' "hole".
      outcoming:  a dictionary `{op: [list_of_holes]}`,
                  where `op` is the layers' "tentacle".
    """
    if within_ops is None:
        within_ops = []
        for op in tf.get_default_graph().get_operations():
            within_ops.append(op)

    within_op_names = [op.name for op in within_ops]
    ops = ge.get_name_scope_ops(within_ops, scope_name)
    incoming, outcoming = {}, {}

    for op in ops:
        src_ops = ge.get_generating_ops(op.inputs)
        src_ops = [o for o in src_ops
                   if o.name in within_op_names]
        dst_ops = ge.get_consuming_ops(op.outputs)
        dst_ops = [o for o in dst_ops
                   if o.name in within_op_names]

        for o in src_ops:
            if not o.name.startswith(scope_name):
                if not op in incoming:
                    incoming[op] = []
                incoming[op].append(o)
        for o in dst_ops:
            if not o.name.startswith(scope_name):
                if not op in outcoming:
                    outcoming[op] = []
                outcoming[op].append(o)

    return incoming, outcoming


def duplicate_layer(layer_name, layer_sgv, branch_name):
    if layer_name[-1] == '/':
        new_layer_name = layer_name[:-1] + branch_name + '/'
    else:
        new_layer_name = layer_name + branch_name + '/'

    replacement_ts = {}
    for op in layer_sgv.inputs:
        replacement_ts[op] = op

    duplicate_sgv, info = ge.copy_with_input_replacements(
        layer_sgv,
        replacement_ts=replacement_ts,
        src_scope=layer_name,
        dst_scope=new_layer_name)
    return info


def reroute_network(outcoming_dict, endpoints, dup_info):
    branch_ops = ge.get_walks_intersection_ops(
        forward_seed_ops=list(outcoming_dict),
        backward_seed_ops=endpoints,
        forward_inclusive=False,
        backward_inclusive=True)

    outputs_to_swap = []
    for op, outputs in outcoming_dict.items():
        outputs_to_swap += [o for o in outputs if o in branch_ops]

    for node in outputs_to_swap:
        orig_inputs = list(node.inputs)
        new_inputs = []
        for ts in orig_inputs:
            new_op = dup_info.transformed(ts.op)
            if new_op is not None:
                new_inputs.extend(new_op.outputs)
            else:
                new_inputs.append(ts)
        ge.reroute_inputs(new_inputs, node)


def unzip(layer_name, branching_scheme, network_ops=None):
    incoming, outcoming = get_tentacles(layer_name, network_ops)
    layer_sgv = ge.make_view_from_scope(layer_name, tf.get_default_graph())

    for branch_name, network_outputs in branching_scheme.items():
        if branch_name == '':
            continue
        info = duplicate_layer(layer_name, layer_sgv, branch_name)
        reroute_network(outcoming, network_outputs, info)
