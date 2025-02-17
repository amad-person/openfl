# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: openfl/protocols/aggregator.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from openfl.protocols import base_pb2 as openfl_dot_protocols_dot_base__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='openfl/protocols/aggregator.proto',
  package='openfl.aggregator',
  syntax='proto3',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n!openfl/protocols/aggregator.proto\x12\x11openfl.aggregator\x1a\x1bopenfl/protocols/base.proto\"o\n\rMessageHeader\x12\x0e\n\x06sender\x18\x01 \x01(\t\x12\x10\n\x08receiver\x18\x02 \x01(\t\x12\x17\n\x0f\x66\x65\x64\x65ration_uuid\x18\x03 \x01(\t\x12#\n\x1bsingle_col_cert_common_name\x18\x04 \x01(\t\"C\n\x0fGetTasksRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .openfl.aggregator.MessageHeader\"S\n\x04Task\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x15\n\rfunction_name\x18\x02 \x01(\t\x12\x11\n\ttask_type\x18\x03 \x01(\t\x12\x13\n\x0b\x61pply_local\x18\x04 \x01(\x08\"\xa4\x01\n\x10GetTasksResponse\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .openfl.aggregator.MessageHeader\x12\x14\n\x0cround_number\x18\x02 \x01(\x05\x12&\n\x05tasks\x18\x03 \x03(\x0b\x32\x17.openfl.aggregator.Task\x12\x12\n\nsleep_time\x18\x04 \x01(\x05\x12\x0c\n\x04quit\x18\x05 \x01(\x08\"\xb1\x01\n\x1aGetAggregatedTensorRequest\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .openfl.aggregator.MessageHeader\x12\x13\n\x0btensor_name\x18\x02 \x01(\t\x12\x14\n\x0cround_number\x18\x03 \x01(\x05\x12\x0e\n\x06report\x18\x04 \x01(\x08\x12\x0c\n\x04tags\x18\x05 \x03(\t\x12\x18\n\x10require_lossless\x18\x06 \x01(\x08\"\x83\x01\n\x1bGetAggregatedTensorResponse\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .openfl.aggregator.MessageHeader\x12\x14\n\x0cround_number\x18\x02 \x01(\x05\x12\x1c\n\x06tensor\x18\x03 \x01(\x0b\x32\x0c.NamedTensor\"\x9a\x01\n\x0bTaskResults\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .openfl.aggregator.MessageHeader\x12\x14\n\x0cround_number\x18\x02 \x01(\x05\x12\x11\n\ttask_name\x18\x03 \x01(\t\x12\x11\n\tdata_size\x18\x04 \x01(\x05\x12\x1d\n\x07tensors\x18\x05 \x03(\x0b\x32\x0c.NamedTensor\"P\n\x1cSendLocalTaskResultsResponse\x12\x30\n\x06header\x18\x01 \x01(\x0b\x32 .openfl.aggregator.MessageHeader\"1\n\x16GetMetricStreamRequest\x12\x17\n\x0f\x65xperiment_name\x18\x01 \x01(\t\"}\n\x17GetMetricStreamResponse\x12\x15\n\rmetric_origin\x18\x01 \x01(\t\x12\x11\n\ttask_name\x18\x02 \x01(\t\x12\x13\n\x0bmetric_name\x18\x03 \x01(\t\x12\x14\n\x0cmetric_value\x18\x04 \x01(\x02\x12\r\n\x05round\x18\x05 \x01(\r\"\xa7\x01\n\x16GetTrainedModelRequest\x12\x17\n\x0f\x65xperiment_name\x18\x02 \x01(\t\x12G\n\nmodel_type\x18\x03 \x01(\x0e\x32\x33.openfl.aggregator.GetTrainedModelRequest.ModelType\"+\n\tModelType\x12\x0e\n\nBEST_MODEL\x10\x00\x12\x0e\n\nLAST_MODEL\x10\x01\"8\n\x14TrainedModelResponse\x12 \n\x0bmodel_proto\x18\x01 \x01(\x0b\x32\x0b.ModelProto\"/\n\x1fGetExperimentDescriptionRequest\x12\x0c\n\x04name\x18\x01 \x01(\t\"N\n GetExperimentDescriptionResponse\x12*\n\nexperiment\x18\x01 \x01(\x0b\x32\x16.ExperimentDescription2\x94\x05\n\nAggregator\x12U\n\x08GetTasks\x12\".openfl.aggregator.GetTasksRequest\x1a#.openfl.aggregator.GetTasksResponse\"\x00\x12v\n\x13GetAggregatedTensor\x12-.openfl.aggregator.GetAggregatedTensorRequest\x1a..openfl.aggregator.GetAggregatedTensorResponse\"\x00\x12X\n\x14SendLocalTaskResults\x12\x0b.DataStream\x1a/.openfl.aggregator.SendLocalTaskResultsResponse\"\x00(\x01\x12l\n\x0fGetMetricStream\x12).openfl.aggregator.GetMetricStreamRequest\x1a*.openfl.aggregator.GetMetricStreamResponse\"\x00\x30\x01\x12g\n\x0fGetTrainedModel\x12).openfl.aggregator.GetTrainedModelRequest\x1a\'.openfl.aggregator.TrainedModelResponse\"\x00\x12\x85\x01\n\x18GetExperimentDescription\x12\x32.openfl.aggregator.GetExperimentDescriptionRequest\x1a\x33.openfl.aggregator.GetExperimentDescriptionResponse\"\x00\x62\x06proto3'
  ,
  dependencies=[openfl_dot_protocols_dot_base__pb2.DESCRIPTOR,])



_GETTRAINEDMODELREQUEST_MODELTYPE = _descriptor.EnumDescriptor(
  name='ModelType',
  full_name='openfl.aggregator.GetTrainedModelRequest.ModelType',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BEST_MODEL', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='LAST_MODEL', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=1375,
  serialized_end=1418,
)
_sym_db.RegisterEnumDescriptor(_GETTRAINEDMODELREQUEST_MODELTYPE)


_MESSAGEHEADER = _descriptor.Descriptor(
  name='MessageHeader',
  full_name='openfl.aggregator.MessageHeader',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='sender', full_name='openfl.aggregator.MessageHeader.sender', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='receiver', full_name='openfl.aggregator.MessageHeader.receiver', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='federation_uuid', full_name='openfl.aggregator.MessageHeader.federation_uuid', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='single_col_cert_common_name', full_name='openfl.aggregator.MessageHeader.single_col_cert_common_name', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=85,
  serialized_end=196,
)


_GETTASKSREQUEST = _descriptor.Descriptor(
  name='GetTasksRequest',
  full_name='openfl.aggregator.GetTasksRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='openfl.aggregator.GetTasksRequest.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=198,
  serialized_end=265,
)


_TASK = _descriptor.Descriptor(
  name='Task',
  full_name='openfl.aggregator.Task',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='openfl.aggregator.Task.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='function_name', full_name='openfl.aggregator.Task.function_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='task_type', full_name='openfl.aggregator.Task.task_type', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='apply_local', full_name='openfl.aggregator.Task.apply_local', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=267,
  serialized_end=350,
)


_GETTASKSRESPONSE = _descriptor.Descriptor(
  name='GetTasksResponse',
  full_name='openfl.aggregator.GetTasksResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='openfl.aggregator.GetTasksResponse.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='round_number', full_name='openfl.aggregator.GetTasksResponse.round_number', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tasks', full_name='openfl.aggregator.GetTasksResponse.tasks', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='sleep_time', full_name='openfl.aggregator.GetTasksResponse.sleep_time', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='quit', full_name='openfl.aggregator.GetTasksResponse.quit', index=4,
      number=5, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=353,
  serialized_end=517,
)


_GETAGGREGATEDTENSORREQUEST = _descriptor.Descriptor(
  name='GetAggregatedTensorRequest',
  full_name='openfl.aggregator.GetAggregatedTensorRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='openfl.aggregator.GetAggregatedTensorRequest.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tensor_name', full_name='openfl.aggregator.GetAggregatedTensorRequest.tensor_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='round_number', full_name='openfl.aggregator.GetAggregatedTensorRequest.round_number', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='report', full_name='openfl.aggregator.GetAggregatedTensorRequest.report', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tags', full_name='openfl.aggregator.GetAggregatedTensorRequest.tags', index=4,
      number=5, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='require_lossless', full_name='openfl.aggregator.GetAggregatedTensorRequest.require_lossless', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=520,
  serialized_end=697,
)


_GETAGGREGATEDTENSORRESPONSE = _descriptor.Descriptor(
  name='GetAggregatedTensorResponse',
  full_name='openfl.aggregator.GetAggregatedTensorResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='openfl.aggregator.GetAggregatedTensorResponse.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='round_number', full_name='openfl.aggregator.GetAggregatedTensorResponse.round_number', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tensor', full_name='openfl.aggregator.GetAggregatedTensorResponse.tensor', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=700,
  serialized_end=831,
)


_TASKRESULTS = _descriptor.Descriptor(
  name='TaskResults',
  full_name='openfl.aggregator.TaskResults',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='openfl.aggregator.TaskResults.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='round_number', full_name='openfl.aggregator.TaskResults.round_number', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='task_name', full_name='openfl.aggregator.TaskResults.task_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='data_size', full_name='openfl.aggregator.TaskResults.data_size', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='tensors', full_name='openfl.aggregator.TaskResults.tensors', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=834,
  serialized_end=988,
)


_SENDLOCALTASKRESULTSRESPONSE = _descriptor.Descriptor(
  name='SendLocalTaskResultsResponse',
  full_name='openfl.aggregator.SendLocalTaskResultsResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='header', full_name='openfl.aggregator.SendLocalTaskResultsResponse.header', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=990,
  serialized_end=1070,
)


_GETMETRICSTREAMREQUEST = _descriptor.Descriptor(
  name='GetMetricStreamRequest',
  full_name='openfl.aggregator.GetMetricStreamRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='experiment_name', full_name='openfl.aggregator.GetMetricStreamRequest.experiment_name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1072,
  serialized_end=1121,
)


_GETMETRICSTREAMRESPONSE = _descriptor.Descriptor(
  name='GetMetricStreamResponse',
  full_name='openfl.aggregator.GetMetricStreamResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='metric_origin', full_name='openfl.aggregator.GetMetricStreamResponse.metric_origin', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='task_name', full_name='openfl.aggregator.GetMetricStreamResponse.task_name', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='metric_name', full_name='openfl.aggregator.GetMetricStreamResponse.metric_name', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='metric_value', full_name='openfl.aggregator.GetMetricStreamResponse.metric_value', index=3,
      number=4, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='round', full_name='openfl.aggregator.GetMetricStreamResponse.round', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1123,
  serialized_end=1248,
)


_GETTRAINEDMODELREQUEST = _descriptor.Descriptor(
  name='GetTrainedModelRequest',
  full_name='openfl.aggregator.GetTrainedModelRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='experiment_name', full_name='openfl.aggregator.GetTrainedModelRequest.experiment_name', index=0,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='model_type', full_name='openfl.aggregator.GetTrainedModelRequest.model_type', index=1,
      number=3, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _GETTRAINEDMODELREQUEST_MODELTYPE,
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1251,
  serialized_end=1418,
)


_TRAINEDMODELRESPONSE = _descriptor.Descriptor(
  name='TrainedModelResponse',
  full_name='openfl.aggregator.TrainedModelResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='model_proto', full_name='openfl.aggregator.TrainedModelResponse.model_proto', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1420,
  serialized_end=1476,
)


_GETEXPERIMENTDESCRIPTIONREQUEST = _descriptor.Descriptor(
  name='GetExperimentDescriptionRequest',
  full_name='openfl.aggregator.GetExperimentDescriptionRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='openfl.aggregator.GetExperimentDescriptionRequest.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1478,
  serialized_end=1525,
)


_GETEXPERIMENTDESCRIPTIONRESPONSE = _descriptor.Descriptor(
  name='GetExperimentDescriptionResponse',
  full_name='openfl.aggregator.GetExperimentDescriptionResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='experiment', full_name='openfl.aggregator.GetExperimentDescriptionResponse.experiment', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1527,
  serialized_end=1605,
)

_GETTASKSREQUEST.fields_by_name['header'].message_type = _MESSAGEHEADER
_GETTASKSRESPONSE.fields_by_name['header'].message_type = _MESSAGEHEADER
_GETTASKSRESPONSE.fields_by_name['tasks'].message_type = _TASK
_GETAGGREGATEDTENSORREQUEST.fields_by_name['header'].message_type = _MESSAGEHEADER
_GETAGGREGATEDTENSORRESPONSE.fields_by_name['header'].message_type = _MESSAGEHEADER
_GETAGGREGATEDTENSORRESPONSE.fields_by_name['tensor'].message_type = openfl_dot_protocols_dot_base__pb2._NAMEDTENSOR
_TASKRESULTS.fields_by_name['header'].message_type = _MESSAGEHEADER
_TASKRESULTS.fields_by_name['tensors'].message_type = openfl_dot_protocols_dot_base__pb2._NAMEDTENSOR
_SENDLOCALTASKRESULTSRESPONSE.fields_by_name['header'].message_type = _MESSAGEHEADER
_GETTRAINEDMODELREQUEST.fields_by_name['model_type'].enum_type = _GETTRAINEDMODELREQUEST_MODELTYPE
_GETTRAINEDMODELREQUEST_MODELTYPE.containing_type = _GETTRAINEDMODELREQUEST
_TRAINEDMODELRESPONSE.fields_by_name['model_proto'].message_type = openfl_dot_protocols_dot_base__pb2._MODELPROTO
_GETEXPERIMENTDESCRIPTIONRESPONSE.fields_by_name['experiment'].message_type = openfl_dot_protocols_dot_base__pb2._EXPERIMENTDESCRIPTION
DESCRIPTOR.message_types_by_name['MessageHeader'] = _MESSAGEHEADER
DESCRIPTOR.message_types_by_name['GetTasksRequest'] = _GETTASKSREQUEST
DESCRIPTOR.message_types_by_name['Task'] = _TASK
DESCRIPTOR.message_types_by_name['GetTasksResponse'] = _GETTASKSRESPONSE
DESCRIPTOR.message_types_by_name['GetAggregatedTensorRequest'] = _GETAGGREGATEDTENSORREQUEST
DESCRIPTOR.message_types_by_name['GetAggregatedTensorResponse'] = _GETAGGREGATEDTENSORRESPONSE
DESCRIPTOR.message_types_by_name['TaskResults'] = _TASKRESULTS
DESCRIPTOR.message_types_by_name['SendLocalTaskResultsResponse'] = _SENDLOCALTASKRESULTSRESPONSE
DESCRIPTOR.message_types_by_name['GetMetricStreamRequest'] = _GETMETRICSTREAMREQUEST
DESCRIPTOR.message_types_by_name['GetMetricStreamResponse'] = _GETMETRICSTREAMRESPONSE
DESCRIPTOR.message_types_by_name['GetTrainedModelRequest'] = _GETTRAINEDMODELREQUEST
DESCRIPTOR.message_types_by_name['TrainedModelResponse'] = _TRAINEDMODELRESPONSE
DESCRIPTOR.message_types_by_name['GetExperimentDescriptionRequest'] = _GETEXPERIMENTDESCRIPTIONREQUEST
DESCRIPTOR.message_types_by_name['GetExperimentDescriptionResponse'] = _GETEXPERIMENTDESCRIPTIONRESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

MessageHeader = _reflection.GeneratedProtocolMessageType('MessageHeader', (_message.Message,), {
  'DESCRIPTOR' : _MESSAGEHEADER,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.MessageHeader)
  })
_sym_db.RegisterMessage(MessageHeader)

GetTasksRequest = _reflection.GeneratedProtocolMessageType('GetTasksRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETTASKSREQUEST,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.GetTasksRequest)
  })
_sym_db.RegisterMessage(GetTasksRequest)

Task = _reflection.GeneratedProtocolMessageType('Task', (_message.Message,), {
  'DESCRIPTOR' : _TASK,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.Task)
  })
_sym_db.RegisterMessage(Task)

GetTasksResponse = _reflection.GeneratedProtocolMessageType('GetTasksResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETTASKSRESPONSE,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.GetTasksResponse)
  })
_sym_db.RegisterMessage(GetTasksResponse)

GetAggregatedTensorRequest = _reflection.GeneratedProtocolMessageType('GetAggregatedTensorRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETAGGREGATEDTENSORREQUEST,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.GetAggregatedTensorRequest)
  })
_sym_db.RegisterMessage(GetAggregatedTensorRequest)

GetAggregatedTensorResponse = _reflection.GeneratedProtocolMessageType('GetAggregatedTensorResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETAGGREGATEDTENSORRESPONSE,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.GetAggregatedTensorResponse)
  })
_sym_db.RegisterMessage(GetAggregatedTensorResponse)

TaskResults = _reflection.GeneratedProtocolMessageType('TaskResults', (_message.Message,), {
  'DESCRIPTOR' : _TASKRESULTS,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.TaskResults)
  })
_sym_db.RegisterMessage(TaskResults)

SendLocalTaskResultsResponse = _reflection.GeneratedProtocolMessageType('SendLocalTaskResultsResponse', (_message.Message,), {
  'DESCRIPTOR' : _SENDLOCALTASKRESULTSRESPONSE,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.SendLocalTaskResultsResponse)
  })
_sym_db.RegisterMessage(SendLocalTaskResultsResponse)

GetMetricStreamRequest = _reflection.GeneratedProtocolMessageType('GetMetricStreamRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETMETRICSTREAMREQUEST,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.GetMetricStreamRequest)
  })
_sym_db.RegisterMessage(GetMetricStreamRequest)

GetMetricStreamResponse = _reflection.GeneratedProtocolMessageType('GetMetricStreamResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETMETRICSTREAMRESPONSE,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.GetMetricStreamResponse)
  })
_sym_db.RegisterMessage(GetMetricStreamResponse)

GetTrainedModelRequest = _reflection.GeneratedProtocolMessageType('GetTrainedModelRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETTRAINEDMODELREQUEST,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.GetTrainedModelRequest)
  })
_sym_db.RegisterMessage(GetTrainedModelRequest)

TrainedModelResponse = _reflection.GeneratedProtocolMessageType('TrainedModelResponse', (_message.Message,), {
  'DESCRIPTOR' : _TRAINEDMODELRESPONSE,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.TrainedModelResponse)
  })
_sym_db.RegisterMessage(TrainedModelResponse)

GetExperimentDescriptionRequest = _reflection.GeneratedProtocolMessageType('GetExperimentDescriptionRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETEXPERIMENTDESCRIPTIONREQUEST,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.GetExperimentDescriptionRequest)
  })
_sym_db.RegisterMessage(GetExperimentDescriptionRequest)

GetExperimentDescriptionResponse = _reflection.GeneratedProtocolMessageType('GetExperimentDescriptionResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETEXPERIMENTDESCRIPTIONRESPONSE,
  '__module__' : 'openfl.protocols.aggregator_pb2'
  # @@protoc_insertion_point(class_scope:openfl.aggregator.GetExperimentDescriptionResponse)
  })
_sym_db.RegisterMessage(GetExperimentDescriptionResponse)



_AGGREGATOR = _descriptor.ServiceDescriptor(
  name='Aggregator',
  full_name='openfl.aggregator.Aggregator',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=1608,
  serialized_end=2268,
  methods=[
  _descriptor.MethodDescriptor(
    name='GetTasks',
    full_name='openfl.aggregator.Aggregator.GetTasks',
    index=0,
    containing_service=None,
    input_type=_GETTASKSREQUEST,
    output_type=_GETTASKSRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetAggregatedTensor',
    full_name='openfl.aggregator.Aggregator.GetAggregatedTensor',
    index=1,
    containing_service=None,
    input_type=_GETAGGREGATEDTENSORREQUEST,
    output_type=_GETAGGREGATEDTENSORRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='SendLocalTaskResults',
    full_name='openfl.aggregator.Aggregator.SendLocalTaskResults',
    index=2,
    containing_service=None,
    input_type=openfl_dot_protocols_dot_base__pb2._DATASTREAM,
    output_type=_SENDLOCALTASKRESULTSRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetMetricStream',
    full_name='openfl.aggregator.Aggregator.GetMetricStream',
    index=3,
    containing_service=None,
    input_type=_GETMETRICSTREAMREQUEST,
    output_type=_GETMETRICSTREAMRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetTrainedModel',
    full_name='openfl.aggregator.Aggregator.GetTrainedModel',
    index=4,
    containing_service=None,
    input_type=_GETTRAINEDMODELREQUEST,
    output_type=_TRAINEDMODELRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
  _descriptor.MethodDescriptor(
    name='GetExperimentDescription',
    full_name='openfl.aggregator.Aggregator.GetExperimentDescription',
    index=5,
    containing_service=None,
    input_type=_GETEXPERIMENTDESCRIPTIONREQUEST,
    output_type=_GETEXPERIMENTDESCRIPTIONRESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_AGGREGATOR)

DESCRIPTOR.services_by_name['Aggregator'] = _AGGREGATOR

# @@protoc_insertion_point(module_scope)
