////////////////////////////////////////////
// Global Variable Definition
// SVG Object
var svgContainer;

// Original Image Link
var links_Img;

// Image Info
var imagefile_org = "";
var img_size_cx = 0;
var img_size_cy = 0;

// Lime Seg Info
var Img_Seg = "";
var Seq_Qty = 0;
var Seg_Data = [];
var Class_Qty = 0;
var Class_Info = [];


function Class_Obj(Class_Name, Feature_Qty)//, Feature_Info_No, Feature_Info_Value)
{
	this.Class_Name = Class_Name;
	this.Feature_Qty = Feature_Qty;
	this.Feature_Info = [];

	this.AddFeature = function(Feature_No, Feature_Value)
	{
		var Feature_Obj = [Feature_No, Feature_Value];
		this.Feature_Info.push(Feature_Obj);
	}
}

// 기존 정보 Clear
function Clear_LIME_Dot()
{
	svgContainer.selectAll("rect").remove();
}

// Dot 그리기..
function Draw_LIME_Dot(Selected_Seg_No)
{
	// 기존 정보 Clear
	Clear_LIME_Dot();

	var Dots = [];
	var Pos_x = 0;
	var Pos_y = 0;
	var Color = 0;
	var Opacity = 1.0;

	for(var sNo in Seg_Data)
	{
		Pos_y = parseInt(sNo / img_size_cx);
		Pos_x = sNo % img_size_cx;

		for(var fNo in Class_Info[Selected_Seg_No].Feature_Info)
		{
			if(Seg_Data[sNo] == Class_Info[Selected_Seg_No].Feature_Info[fNo][0])
			{
				Dots.push({"x":Pos_x, "y":Pos_y, "color": 255 - Class_Info[Selected_Seg_No].Feature_Info[fNo][0]*10});
				break;
			}
		}
	}

	var links = svgContainer.selectAll(".link")
		.data(Dots)
		.enter();//.append("g");

	var lines = links.append("rect")
		.attr("x", function(d) { return d.x;})
		.attr("y", function(d) { return d.y;})
		.attr("width", 2)
		.attr("height", 2)
		.attr("fill", function(d) { return "rgba("+d.color+",0,0,"+Opacity+")";});
};

function init() {
	var margin = {
			top: 50,
			right: 50,
			bottom: 50,
			left: 50
		},
	width = 1200 - margin.left - margin.right,
	height = 900 - margin.top - margin.bottom;

	svgContainer = d3.select("body").append("svg")
		.attr("x", margin.left)
		.attr("y", margin.top)
		.attr("width", width + margin.left + margin.right)
		.attr("height", height + margin.top + margin.bottom)
		.append("g")
		.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	d3.json("ImageRule.json", function(error, data) {
		if (error) throw error;

		//-------------------------------------------------------------
		// Image 기본 정보 가져오기
		imagefile_org = "./" + data.img_file_org;
		img_size_cx = data.img_size_cx;
		img_size_cy = data.img_size_cy;

		//-------------------------------------------------------------
		// Segmentation 정보 읽기

		Img_Seg = data.LIME.img_file_seg;
		Seq_Qty = data.LIME.Seq_Qty;
		for (var s in data.LIME.Seg_Data)
			Seg_Data.push(data.LIME.Seg_Data[s]);

		Class_Qty = data.LIME.Class_Qty;
		Class_Info = [];

		for (var c in data.LIME.Class_Info)
		{
			var class_object = new Class_Obj(data.LIME.Class_Info[c].Class_Name, data.LIME.Class_Info[c].Feature_Qty);
			for(var f in data.LIME.Class_Info[c].Feature_Info)
			{
				//console.log(data.LIME.Class_Info[c].Feature_Info[f][0], data.LIME.Class_Info[c].Feature_Info[f][1]);
				class_object.AddFeature(data.LIME.Class_Info[c].Feature_Info[f][0], data.LIME.Class_Info[c].Feature_Info[f][1]);
			}
			Class_Info.push(class_object);
		}
		//-------------------------------------------------------------
		// Image 출력
		var Imgs = [];
		Imgs.push({"src":imagefile_org, "width":img_size_cx, "height":img_size_cy});

		links_Img = svgContainer.selectAll(".link")
			.data(Imgs)
			.enter().append("g");

		var Imgs = links_Img.append("image")
			.attr("xlink:href", function(d) { return d.src;})
			.attr("x", 0)
			.attr("y", 0)
			.attr("width", function(d) { return d.width;})
			.attr("height", function(d) { return d.height; });

		//-------------------------------------------------------------
		// Segmentation 출력
		var Selected_Seg_No = 1;
		//Draw_LIME_Dot(Selected_Seg_No);
	})
}
