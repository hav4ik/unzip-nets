import tensorflow as tf
import tensorflow.contrib.graph_editor as ge
from tensorflow.core.framework import variable_pb2
from keras import backend as K


def _get_tentacles(scope_name, within_ops=None):
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
                if op not in incoming:
                    incoming[op] = []
                incoming[op].append(o)
        for o in dst_ops:
            if not o.name.startswith(scope_name):
                if op not in outcoming:
                    outcoming[op] = []
                outcoming[op].append(o)

    return incoming, outcoming


def _duplicate_layer(layer_name,
                     layer_sgv,
                     branch_name,
                     add_to_collections=True):
    """Duplicates a network layer, while preserving connections.

    Args:
      layer_name:         a layer is identified by its name scope
      layer_sgv:          SubgraphView (see tf.contrib.graph_editor)
      branch_name:        the duplicate is "layer_name + branch_name"
      add_to_collections: add duplicate vars to the same collections

    Returns:
      info:            see ret vals of `tf.contrib.graph_editor.copy`
      var_duplication: a list of tuples (var, dup_of_var)
    """

    if layer_name[-1] == '/':
        new_layer_name = layer_name[:-1] + branch_name + '/'
    else:
        new_layer_name = layer_name + branch_name

    replacement_ts = {}
    for op in layer_sgv.inputs:
        replacement_ts[op] = op

    duplicate_sgv, info = ge.copy_with_input_replacements(
        layer_sgv,
        replacement_ts=replacement_ts,
        src_scope=layer_name,
        dst_scope=new_layer_name)

    var_duplication = []
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if layer_name not in v.name:
            continue
        vproto = v.to_proto()
        new_vardef = variable_pb2.VariableDef()
        for field, val in vproto.ListFields():
            if isinstance(val, str):
                new_val = val.replace(layer_name, new_layer_name)
            else:
                new_val = val
            setattr(new_vardef, field.name, new_val)
        new_var = tf.Variable(variable_def=new_vardef)
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, new_var)
        var_duplication.append((v, new_var))

        if add_to_collections:
            for k in tf.get_default_graph().get_all_collection_keys():
                collection = tf.get_collection(k)
                if v in collection and new_var not in collection:
                    tf.add_to_collection(k, new_var)

    return info, var_duplication


def _reroute_network(outcoming_dict, endpoints, dup_info):
    """
    Called after _duplicate_layer. Re-route the paths from layers' outputs
    to the network's endpoints to the duplicate layer.

    Args:
      outcoming_dict: a dict {op: [outputs]} of original layers' outcoming
                      nodes; only ops from the layer and outputs outside
                      the layer are considered.
      endpoints:      network's endpoints (outputs to task-specific heads)
      dup_info:       the `info` ret val of _duplicate_layer.
    """
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


def do_branching(layer_name, branching_scheme, network_ops=None):
    """
    Split a given layer into branches in such a way, that the sub-graph with
    name scope "layer_name" now outputs to branching_scheme[''], and sub-graph
    with name scope "layer_name + x" now outputs to branching_scheme[x].

    Args:
      layer_name:       a layer is identified by the common scope of its ops
      branching_scheme: a dict {'branch_suffix': [network_endpoints]}
      network_ops:      list of ops in the network

    Returns:
      duplicates:       a list of tuples of tf.Variables (var, dup_of_var)
    """
    incoming, outcoming = _get_tentacles(layer_name, network_ops)
    layer_sgv = ge.make_view_from_scope(layer_name, tf.get_default_graph())

    duplicates = []
    for branch_name, network_outputs in branching_scheme.items():
        if branch_name == '':
            continue
        info, dups = _duplicate_layer(layer_name, layer_sgv, branch_name)
        _reroute_network(outcoming, network_outputs, info)
        duplicates.extend(dups)

    return duplicates


def unzip(sess,
          network_ops,
          layer_name,
          branching_scheme,
          session_prep=None,
          saver=None,
          saver_scope='save'):
    """Performs unzipping on netwokr's graph while preserving the weights

    Args:
      sess:             current session; this is going to be closed and f*cked.
      network_ops:      list of ops in the network.
      layer_name:       name of the layer to perform unzipping.
      branching_scheme: a dict {'branch_suffix': [network_endpoints]}
      session_prep:     function that makes a new session (for customized sess)
      saver:            model's saver; if None, tf.train.Saver() is called.
      saver_scope:      if `saver` != None & the savers have non-standard scope

    Returns:
      sess:             a current session that holds the new graph and weights.
      saver:            a saver for the model after surgery.
    """

    if saver is None:
        pre_surgery_saver = tf.train.Saver(name=saver_scope)
    else:
        pre_surgery_saver = saver
    pre_surgery_saver.save(sess, '/tmp/pre_surgery')
    sess.close()

    duplicate_var_pairs = do_branching(
            layer_name, branching_scheme, network_ops)

    if session_prep is None:
        sess = tf.Session()
    else:
        sess = session_prep()
    K.set_session(sess)

    pre_surgery_saver.restore(sess, '/tmp/pre_surgery')

    for var, new_var in duplicate_var_pairs:
        new_var.load(var.eval(sess), sess)
    post_surgery_saver = tf.train.Saver()
    post_surgery_saver.save(sess, '/tmp/post_surgery', write_meta_graph=False)

    non_saver_nodes = []
    for node in tf.get_default_graph().as_graph_def().node:
        if not node.name.startswith(saver_scope):
            non_saver_nodes.append(node.name)
    no_saver_graphdef = tf.graph_util.extract_sub_graph(
        tf.get_default_graph().as_graph_def(), non_saver_nodes)
    tf.train.export_meta_graph(
            '/tmp/full_saver.meta', graph_def=no_saver_graphdef)

    sess.close()
    K.clear_session()
    tf.reset_default_graph()

    if session_prep is None:
        sess = tf.Session()
    else:
        sess = session_prep()
    K.set_session(sess)

    full_saver = tf.train.import_meta_graph('/tmp/full_saver.meta')
    full_saver.restore(sess, '/tmp/post_surgery')

    return sess, full_saver
