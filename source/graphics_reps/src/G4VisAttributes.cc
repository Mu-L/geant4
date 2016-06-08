// This code implementation is the intellectual property of
// the GEANT4 collaboration.
//
// By copying, distributing or modifying the Program (or any work
// based on the Program) you indicate your acceptance of this statement,
// and all its terms.
//
// $Id: G4VisAttributes.cc,v 1.5 1999/12/15 14:50:37 gunter Exp $
// GEANT4 tag $Name: geant4-02-00 $
//
// 
// John Allison  23rd October 1996

#include "G4VisAttributes.hh"

G4VisAttributes::G4VisAttributes ():
fVisible            (true),
fDaughtersInvisible (false),
fColour             (G4Colour ()),
fLineStyle          (unbroken),
fLineWidth          (1.),
fForceDrawingStyle  (false)
{}

G4VisAttributes::G4VisAttributes (G4bool visibility):
fVisible            (visibility),
fDaughtersInvisible (false),
fColour             (G4Colour ()),
fLineStyle          (unbroken),
fLineWidth          (1.),
fForceDrawingStyle  (false)
{}

G4VisAttributes::G4VisAttributes (const G4Colour& colour):
fVisible            (true),
fDaughtersInvisible (false),
fColour             (colour),
fLineStyle          (unbroken),
fLineWidth          (1.),
fForceDrawingStyle ( false)
{}

G4VisAttributes::G4VisAttributes (G4bool visibility,
				  const G4Colour& colour):
fVisible            (visibility),
fDaughtersInvisible (false),
fColour             (colour),
fLineStyle          (unbroken),
fLineWidth          (1.),
fForceDrawingStyle  (false)
{}

const G4VisAttributes  G4VisAttributes::Invisible = G4VisAttributes (false);

G4std::ostream& operator << (G4std::ostream& os, const G4VisAttributes& a) {
  
  os << "G4VisAttributes: ";
  if (&a){
    if (!a.fVisible) os << "in";
    os << "visible, daughters ";
    if (a.fDaughtersInvisible) os << "in";
    os << "visible, colour: " << a.fColour;
    os << "\n  linestyle: ";
    switch (a.fLineStyle) {
    case G4VisAttributes::unbroken:
      os << "solid"; break;
    case G4VisAttributes::dashed:
      os << "dashed"; break;
    case G4VisAttributes::dotted: os << "dotted"; break;
    default: os << "unrecognised"; break;
    }
    os << ", line width: " << a.fLineWidth;
    os << "\n  drawing style ";
    if (a.fForceDrawingStyle) {
      os << "forced to: ";
      switch (a.fForcedStyle) {
      case G4VisAttributes::wireframe:
	os << "wireframe"; break;
      case G4VisAttributes::solid:
	os << "solid"; break;
      default: os << "unrecognised"; break;
      }
    }
    else {
      os << "unforced";
    }
  } 
  else os << " The pointer is zero ";
  return os;

}

G4bool G4VisAttributes::operator != (const G4VisAttributes& a) const {

  if (
      (fVisible            != a.fVisible)            ||
      (fDaughtersInvisible != a.fDaughtersInvisible) ||
      (fColour             != a.fColour)             ||
      (fLineStyle          != a.fLineStyle)          ||
      (fLineWidth          != a.fLineWidth)          ||
      (fForceDrawingStyle  != a.fForceDrawingStyle)
      )
    return true;

  if (fForceDrawingStyle) {
    if (fForcedStyle != a.fForcedStyle) return true;
  }

  return false;
}

G4bool G4VisAttributes::operator == (const G4VisAttributes& a) const {
  return !(G4VisAttributes::operator != (a));
}
