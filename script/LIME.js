////////////////////////////////////////////
// Global Variable Definition
// SVG Object
var svgContainer;
//Image_dsp
// Original Image Link
var links_Img;

// Button Objec
var BtnContainer;

// Model Image Link
var ImageName;

// Image Info
var imagefile_org = "";
var imagefile_seg = "";
var img_size_cx = 0;
var img_size_cy = 0;

// Lime Seg Info
var Img_Seg = "";
var Seq_Qty = 0;
var Seg_Data = [];
var Class_Qty = 0;
var Class_Info = [];

var Selected_Class_No = -1;

var color_on = "rgb(123, 255, 116)";
var color_off = "rgb(223, 240, 216)";
var color_selected = "orange";


var color_on_clear = "rgb(173, 255, 166)";//"orange";
var color_off_clear = "rgb(200,200,200)";

var ShowFeatureQty = 0;
var bShowSegment = false;



function Class_Obj(Class_Name, Class_Predict, Class_No, Feature_Qty)//, Feature_Info_No, Feature_Info_Value)
{
	this.Class_Name = Class_Name;
	this.Class_Predict = Class_Predict;
	this.Class_No = Class_No;
	this.Feature_Qty = Feature_Qty;
	this.Feature_Info = [];

	this.AddFeature = function(Feature_No, Feature_Value)
	{
		var Feature_Obj = [Feature_No, Feature_Value];
		this.Feature_Info.push(Feature_Obj);
	}
}

// ���� ���� Clear
function Clear_LIME_Dot()
{
	if(svgContainer)
		svgContainer.selectAll("#Feature_Dot").remove();
	Selected_Class_No = -1;
}
function Clear_LIME_Btn()
{
	if(svgContainer)
		svgContainer.selectAll("#Feature_Image").remove();

	if(BtnContainer)
	{
		BtnContainer.selectAll("#Feature_Btn_Text").remove();
		BtnContainer.selectAll("#Feature_Btn").remove();
	}

	Selected_Class_No = -1;
}


function Draw_LIME_Btn()
{
	if(BtnContainer)
	{
		BtnContainer.selectAll("#Feature_Btn_Text").remove();
		BtnContainer.selectAll("#Feature_Btn").remove();
	}

	//-------------------------------------------------------------
	// Button ���
	BtnContainer = d3.select("#Class_Type");
/*
		BtnContainer = d3.select("#Class_Type");

		var y1 = 0;
		var gapText = 10;
		var TextAry = [];
		var TextTmp = "";

		for(var cNo in Class_Info)
		{
			TextTmp = "["+Class_Info[cNo].Class_No + "] ("+Class_Info[cNo].Class_Predict + "%) " + Class_Info[cNo].Class_Name;
			TextAry.push(TextTmp);
			console.log(TextTmp);
		}
		TextAry.push("Clear");


		var links = BtnContainer.selectAll(".link")
			.data(TextAry)
			.enter();

		links.append("rect")
			.attr("id", "Feature_Btn")
			.attr("x", 0)
			.attr("y", function(d, i) {return i*25+20;})
			.attr("width", 250)
			.attr("height", 20)
			.on("mouseover", function(d) { d3.select(this).attr("fill", color_on);})
			.on("mouseleave", function(d, i) { d3.select(this).attr("fill", (i<Class_Info.length)?color_off : color_off_clear); })
			.on("click", function(d,i) {return Draw_LIME_Dot(i);})
			.attr("fill", function(d, i) { return (i<Class_Info.length)? color_off : color_off_clear; });

		links.append("text")
			.attr("id", "Feature_Btn_Text")
			.attr("x", 10 )
			.attr("y",  function(d, i) {return i*25+35;} )
			.text(function(d) { return d; })
			.on("click", function(d,i) {return Draw_LIME_Dot(i);})
			.style("color", "black");
*/

	var y1 = 0;
	var gapText = 10;
	var TextAry = [];
	var TextTmp = "";

	for(var cNo in Class_Info)
	{
		TextTmp = "["+Class_Info[cNo].Class_No + "] ("+Class_Info[cNo].Class_Predict + "%) " + Class_Info[cNo].Class_Name;
		TextAry.push(TextTmp);
		//console.log(TextTmp);
	}
	TextAry.push("Clear");


	var links = BtnContainer.selectAll(".link")
		.data(TextAry)
		.enter();

	links.append("rect")
		.attr("id", "Feature_Btn")
		.attr("x", 0)
		.attr("y", function(d, i) {return i*25+20;})
		.attr("width", 350)
		.attr("height", 20)
		.on("mouseover", function(d) { d3.select(this).attr("fill", color_on);})
		.on("mouseleave", function(d, i) { d3.select(this).attr("fill", (i<Class_Info.length)? ((i==Selected_Class_No)? color_selected:color_off) : color_off_clear); })
		.on("click", function(d,i) {return Draw_LIME_Dot(i);})
		.attr("fill", function(d, i) { return (i<Class_Info.length)? ((i==Selected_Class_No)? color_selected:color_off) : color_off_clear; });

	links.append("text")
		.attr("id", "Feature_Btn_Text")
		.attr("x", 10 )
		.attr("y",  function(d, i) {return i*25+35;} )
		.text(function(d) { return d; })
		.on("click", function(d,i) {return Draw_LIME_Dot(i);})
		.style("color", "black");

	console.log(color_selected);
}

function ShowSegment()
{
	bShowSegment = !bShowSegment;
	RefreshImage();
	Draw_LIME_Dot(Selected_Class_No);
}

function RefreshImage()
{
	svgContainer.selectAll("#Feature_Image").remove();
	//console.log("imagefile_org" + imagefile_org );
	//console.log("imagefile_seg" + imagefile_seg );

	//-------------------------------------------------------------
	// Image ���
	var Imgs = [];
	if(bShowSegment)
		Imgs.push({"src":imagefile_seg, "width":img_size_cx, "height":img_size_cy});
	else
		Imgs.push({"src":imagefile_org, "width":img_size_cx, "height":img_size_cy});

	links_Img = svgContainer.selectAll(".link")
		.data(Imgs)
		.enter().append("g");

	var Imgs = links_Img.append("image")
		.attr("id", "Feature_Image")
		.attr("xlink:href", function(d) { return d.src;})
		.attr("x", 0)
		.attr("y", 0)
		.attr("width", function(d) { return d.width;})
		.attr("height", function(d) { return d.height; });

	//console.log("imagefile_org" + imagefile_org );
}




// Model Image Change
function ModelChange(ImgName)
{
	if(ImageName == ImgName)
	{
		Clear_LIME_Dot();
		return;
	}

	ImageName = ImgName;

	svgContainer = d3.select("#Image_dsp");

	var JsonFile = "./static/Model/"+ImgName + "/LIME.json";
//    print(JsonFile)

	d3.json(JsonFile, function(error, data) {

		if (error) throw error;

		//-------------------------------------------------------------
		// Image �⺻ ���� ��������
		imagefile_org = "./static/Model/" + ImgName + "/" + data.img_file_org;
		img_size_cx = data.img_size_cx;
		img_size_cy = data.img_size_cy;

		//-------------------------------------------------------------
		// Segmentation ���� �б�
		imagefile_seg = "./static/Model/" + ImgName + "/" + data.LIME.img_file_seg;
		Seq_Qty = data.LIME.Seq_Qty;

		Seg_Data = [];

		for (var sRow in data.LIME.Seg_Data)
		{
			var sCols = [];
			for (var sCol in data.LIME.Seg_Data[sRow])
			{
				sCols.push(data.LIME.Seg_Data[sRow][sCol]);
			}
			Seg_Data.push(sCols);
		}

		Class_Qty = data.LIME.Class_Qty;
		Class_Info = [];

		for (var c in data.LIME.Class_Info)
		{
			var class_object = new Class_Obj(data.LIME.Class_Info[c].Class_Name, data.LIME.Class_Info[c].Class_Predict, data.LIME.Class_Info[c].Class_No, data.LIME.Class_Info[c].Feature_Qty);
			for(var f in data.LIME.Class_Info[c].Feature_Info)
			{
				class_object.AddFeature(data.LIME.Class_Info[c].Feature_Info[f][0], data.LIME.Class_Info[c].Feature_Info[f][1]);
			}
			Class_Info.push(class_object);
		}

		//-------------------------------------------------------------
		// Image ���
		RefreshImage();
		//-------------------------------------------------------------
		// Button ���

		Draw_LIME_Btn();
		//Draw_LIME_Dot(Selected_Seg_No);
	});
}

function FeatureQty(FeatureQ)
{
	ShowFeatureQty = FeatureQ - 1;
	Draw_LIME_Dot(Selected_Class_No);
}


//-------------------------------------------------------------
// Segmentation Dot ���
function Draw_LIME_Dot(Selected_Seg_No)
{
	// ���� ���� Clear
	Clear_LIME_Dot();

	if(Selected_Seg_No >= Class_Info.length || Selected_Seg_No<0)
		return;

	Selected_Class_No = Selected_Seg_No;
	Draw_LIME_Btn();

	var Dots = [];
	var Pos_x = 0;
	var Pos_y = 0;
	var Color = 0;
	var Opacity = 0.5;

	sNo = 0;
	var sColor;
	for(var row in Seg_Data)
	{
		Pos_x = 0;
		for(var col in Seg_Data[row])
		{
			for(var fNo in Class_Info[Selected_Seg_No].Feature_Info)
			{
				if(fNo > ShowFeatureQty)
					break;

				if(Seg_Data[row][col] == Class_Info[Selected_Seg_No].Feature_Info[fNo][0])
				{
					if(Class_Info[Selected_Seg_No].Feature_Info[fNo][1] > 0.0)
						sColor = "rgba(0,255,0,"+Opacity+")";
					else
						sColor = "rgba(255,0,0,"+Opacity+")";

					Dots.push({"x":Pos_x, "y":Pos_y, "color": sColor});
					break;
				}
			}
			sNo++;
			Pos_x++;
		}
		Pos_y++;

	}

	var links = svgContainer.selectAll(".link")
		.data(Dots)
		.enter();//.append("g");

	var lines = links.append("rect")
		.attr("id", "Feature_Dot")
		.attr("x", function(d) { return d.x;})
		.attr("y", function(d) { return d.y;})
		.attr("width", 1)
		.attr("height", 1)
		.attr("fill", function(d) { return d.color;});
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

	ImgName = "Guitar";

	ModelChange(ImgName);
}
