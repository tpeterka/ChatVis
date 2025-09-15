from paraview.simple import *

# read dataset
reader = OpenDataFile('disk_out_ref.ex2')

# print values
pd = reader.PointData
for ai in pd.values():
   print(ai.GetName(), ai.GetNumberOfComponents(), end=" ")
   for i in range(ai.GetNumberOfComponents()):
      print(ai.GetRange(i), end=" ")
   print()

# color by pressure "Pres"
# readerRep = GetRepresentation()
renderView1 = GetActiveViewOrCreate('RenderView')
dataDisplay = Show(reader, renderView1)
ColorBy(dataDisplay, ("POINTS", "Pres"))
UpdateScalarBars()
Show()
Render()

# create a line plot
plot = PlotOverLine()
plot.Point1 = [0, 0, 0]
plot.Point2 = [0, 0, 10]
writer = CreateWriter("line-plot.csv")
writer.UpdatePipeline()

# save the plot
plotView = FindViewOrCreate("MyView", viewtype="XYChartView")
Show(plot)
Render()
SaveScreenshot("line-plot.png")

