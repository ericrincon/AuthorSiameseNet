require 'torch'
require 'model'
cmd = torch.CmdLine()
cmd:option('-units', 10, 'Hidden layer units')
cmd:parse(arg)

print(cmd['units'])
